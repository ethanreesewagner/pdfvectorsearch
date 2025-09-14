from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import dotenv_values
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool

@tool
def create_embeddings(file_path: str) -> str:
    """Creates and stores embeddings for a given PDF file in Pinecone. The input should be the file_path of the PDF document."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)

    texts = [chunk.page_content for chunk in chunks]

    config=dotenv_values(".env")
    client = OpenAI(api_key=config["OPENAI_API_KEY"])
    pc = Pinecone(api_key=config["PINECONE_API_KEY"])

    response = client.embeddings.create(input=texts,
    model="text-embedding-3-small", dimensions=512)

    embeddings = [data.embedding for data in response.data]

    index = pc.Index("testingvectors")

    vectors = []
    for i, embedding in enumerate(embeddings):
        vectors.append({
            "id": f"chunk_{i}",
            "values": embedding,
            "metadata": {"title": file_path, "text": texts[i], "page": chunks[i].metadata["page"]}
        })

    upsert_response = index.upsert(
        vectors=vectors,
        namespace="example-namespace"
    )

    return "Embeddings created successfully." + str(upsert_response) 