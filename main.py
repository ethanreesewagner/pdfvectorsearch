from pinecone import Pinecone
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import dotenv_values
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

file_path = "/Users/wowagner/pdfvectorsearch/abridged.pdf"
loader = PyPDFLoader(file_path)
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)
chunks = text_splitter.split_documents(documents)

# Extract text from chunks
texts = [chunk.page_content for chunk in chunks]

config=dotenv_values(".env")
client = OpenAI(api_key=config["OPENAI_API_KEY"])
pc = Pinecone(api_key=config["PINECONE_API_KEY"])

response = client.embeddings.create(input=texts,
model="text-embedding-3-small", dimensions=512)

# Get all embeddings
embeddings = [data.embedding for data in response.data]

index = pc.Index("testingvectors")

# Upsert all chunks with their embeddings
vectors = []
for i, embedding in enumerate(embeddings):
    vectors.append({
        "id": f"chunk_{i}",
        "values": embedding,
        "metadata": {"title": file_path}
    })

upsert_response = index.upsert(
    vectors=vectors,
    namespace="example-namespace"
)