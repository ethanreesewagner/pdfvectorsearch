from flask import Flask, request
from flask.cli import routes_command
from flask.typing import RouteCallable
from langchain_community.document_loaders.pdf import PyPDFLoader
import tempfile
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import dotenv_values
from openai import OpenAI
from agent_functions import process_user_input

path=""

app = Flask(__name__)

@app.route('/agent', methods=['POST'])
def send_to_agent():
    data = request.get_json()
    message=data["message"]
    return process_user_input(message+f" The path is {path}")

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files["file"]
    # Read the file into memory as bytes
    file_bytes = file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
        temp_pdf_file.write(file_bytes)
        temp_pdf_path = temp_pdf_file.name
    # Initialize PyPDFLoader with the path to the temporary file
    loader = PyPDFLoader(temp_pdf_path)
    # Properly load the document using the loader
    docs = loader.load()
    #WARNING RETURNS ONLY FIRST PAGE MUST BE FIXED LATER
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
    )
    chunks = text_splitter.split_documents(docs)

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
            "metadata": {"title": temp_pdf_path, "text": texts[i], "page": chunks[i].metadata["page"]}
        })

    upsert_response = index.upsert(
        vectors=vectors,
        namespace="example-namespace"
    )
    # Clean up the temporary file
    global path
    path=temp_pdf_path
    os.unlink(temp_pdf_path)
    # Convert upsert_response to dict if possible for serialization
    # Ensure the response is JSON serializable
    from flask import jsonify

    def make_serializable(obj):
        # Recursively convert objects to serializable types
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        elif hasattr(obj, "to_dict"):
            return make_serializable(obj.to_dict())
        elif hasattr(obj, "__dict__"):
            return make_serializable(vars(obj))
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            return str(obj)

    if docs:
        # Instead of returning a descriptor, return the actual vectors data that was upserted
        # Ensure all objects are JSON serializable
        serializable_vectors = make_serializable(vectors)
        return jsonify({"vectors": serializable_vectors})
    else:
        return "No content found."

from flask import send_from_directory

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)