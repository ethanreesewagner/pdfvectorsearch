from flask import Flask, request
import mimetypes
from flask.cli import routes_command
from flask.typing import RouteCallable
from langchain_core.document_loaders.blob_loaders import BlobLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
import agent_functions

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files["file"]
    # Read the file into memory as bytes
    file_bytes = file.read()
    # Guess the mime type using the filename
    mime_type, _ = mimetypes.guess_type(file.filename)
    # Create a Blob object from the in-memory bytes
    blob = BlobLoader.from_data(
        data=file_bytes,
        path=file.filename,
        mime_type=mime_type
    )
    # Use the PyPDFLoader with the blob (in-memory, no disk write)
    loader = PyPDFLoader(blob_loader=blob)
    # Properly load the document using the loader
    docs = loader.load()
    # For demonstration, return the first page's content (or all as needed)
    return docs[0].page_content if docs else "No content found."

if __name__ == '__main__':
    app.run(debug=True)