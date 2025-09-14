from flask import Flask, request
import mimetypes
from flask.cli import routes_command
from flask.typing import RouteCallable
from langchain_community.document_loaders.blob_loaders import Blob
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
    # blob = Blob.from_data(
    #     data=file_bytes,
    #     path=file.filename,
    #     mime_type=mime_type
    # )
    # Use the PyPDFLoader with the blob (in-memory, no disk write)
    # loader = PyPDFLoader(blob=blob)

    # Create a temporary file to store the uploaded PDF
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
        temp_pdf_file.write(file_bytes)
        temp_pdf_path = temp_pdf_file.name

    # Initialize PyPDFLoader with the path to the temporary file
    loader = PyPDFLoader(temp_pdf_path)

    # Properly load the document using the loader
    docs = loader.load()

    # Clean up the temporary file
    import os
    os.unlink(temp_pdf_path)
    # For demonstration, return the first page's content (or all as needed)
    return docs[0].page_content if docs else "No content found."

if __name__ == '__main__':
    app.run(debug=True)