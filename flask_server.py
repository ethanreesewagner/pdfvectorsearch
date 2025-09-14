from flask import Flask, request
from flask.cli import routes_command
from flask.typing import RouteCallable
from langchain_community.document_loaders.pdf import PyPDFLoader
import tempfile
import os
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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
        temp_pdf_file.write(file_bytes)
        temp_pdf_path = temp_pdf_file.name
    # Initialize PyPDFLoader with the path to the temporary file
    loader = PyPDFLoader(temp_pdf_path)
    # Properly load the document using the loader
    docs = loader.load()
    # Clean up the temporary file
    os.unlink(temp_pdf_path)
    #WARNING RETURNS ONLY FIRST PAGE MUST BE FIXED LATER
    return docs[0].page_content if docs else "No content found."

if __name__ == '__main__':
    app.run(debug=True)