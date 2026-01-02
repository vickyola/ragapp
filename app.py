# Kleines RAG Projekt mit Chroma Vector Database

# Standard Library Imports
import json
import os
import re
import sys
from pathlib import Path
import urllib.parse

# Third-Party Library Imports
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash, g
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
import chromadb

load_dotenv()

app = Flask(__name__)

# Change here!
app.config['UPLOAD_FOLDER'] = "/chat/uploads"

# OpenAI API Key 
openapi_key =  os.getenv("OPENAI_API_KEY")
rolle = "Du bist ein freundlicher Assistent. Du recherchierst Informationen und Beantwortest Fragen. Du Antwortest nur auf Basis der dir bereit stehenden Informationen und erfindest nichts dazu und Antwortest im JSON Format."
# embedding_function
openai_ef = OpenAIEmbeddings(api_key=openapi_key , model="text-embedding-ada-002")

# # Chroma Database
def clean_vector_store(db_directory="./chroma_vectorstore", collection_name="Contextdaten", embedding_function=None):
    """
    Cleans up the Chroma vector store by removing all documents in the specified collection.
    """
    try:
        # Initialize the vector database
        db = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=db_directory
        )
        # Get all documents in the collection
        all_docs = db.get()
        print(f"Total documents before cleaning: {len(all_docs['documents'])}")
        print(f"Document IDs: {all_docs['ids']}")
        print(f"Metadata: {all_docs['metadatas']}")
        # Delete all documents
        db.delete_collection()
        print(f"Collection '{collection_name}' has been deleted.")
    except Exception as e:
        print(f"Error during vector store clean-up: {e}")


@app.route('/success/<path:file_path>')
def vectordb(file_path, db_directory="./chroma_vectorstore"):
    try:
        file_path = urllib.parse.unquote(file_path)

        # Ensure the file path is absolute by resolving it relative to the app's base directory
        if not file_path.startswith("/"):
            file_path = "/" + file_path

        absolute_file_path = Path(file_path)
         # Combine base directory with file_path
        print(f"Absolute file path: {absolute_file_path}")

        # Check if the file exists and is a file
        if not absolute_file_path.is_file():
            return jsonify({'error': f"File hier not found: {absolute_file_path}"}), 404

        embeddings = openai_ef

        # Load the Chroma vector store
        db = Chroma( collection_name="Contextdaten", embedding_function=embeddings, persist_directory=db_directory)

        # Extract file name and check if it exists in the vector store
        file_name = os.path.basename(absolute_file_path)
       
        print(f"Adding document '{file_name}' to the vector store...")

        # Load the document and split it into chunks
       # loader = PyPDFLoader(file_path)

        loader = PyPDFLoader(str(absolute_file_path))


        documents = loader.load()
        # try differnt Text splitters
        text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        docs = text_splitter.split_documents(documents)

        # Add metadata and store chunks in the vector store
        for i, doc in enumerate(docs):
            doc.metadata["source"] = file_name
            doc.metadata["chunk_index"] = i
            print(f"Chunk {i}:")
            print(f"  Content: {doc.page_content[:100]}")  # Check content
            print(f"  Metadata: {doc.metadata}")         # Check metadata
            print(f"Metadata for Chunk {i}: {doc.metadata}")


        db.add_documents(docs)

        print(f"Document '{file_name}' has been added to the vector store.")

        return jsonify({'message': f"Document '{file_name}' processed successfully."})
    except Exception as e:

        print(f"Error during processing: {e}")
        return jsonify({'error': str(e)}), 500

def similarity_search(prompt, db_directory="./chroma_vectorstore"):
    """
    Performs a similarity search using a user-provided prompt.
    """
    try:
        # Initialize the vector database
        embeddings = openai_ef  # Replace with your embedding function
        db = Chroma( collection_name="Contextdaten", embedding_function=openai_ef, persist_directory=db_directory)

        # Perform the similarity search
        print(f"Performing similarity search for prompt: '{prompt}'")
               
        # Embed prompt
        prompt_embedding = embeddings.embed_query(prompt)
        results = db.similarity_search_by_vector_with_relevance_scores(prompt_embedding, k=3)

        
        for doc, score in results:
            print(f"Page Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
            print(f"Score: {score}")


        contents = [doc.page_content for doc, _ in results]

    # Option 1: return as single string (recommended for LLM context)
        return "\n\n---\n\n".join(contents)


    except Exception as e:
        print(f"Error during similarity search: {e}")
        return None
    
def get_completion(prompt):
    print(prompt) #debug

    search_result= similarity_search(prompt)

    client = OpenAI()
    response = client.chat.completions.create(
                    model = "gpt-4o",
                    response_format={ "type": "json_object" },
                    temperature = 0.3,
                    messages=[
                    {"role": "system", "content": rolle },
                        {"role": "user", "content": "Beantworte diese user query: " + prompt + " mit dem folgendem context: " + str(search_result)+ "wenn du die Antwort auf die Frage nicht im context findest erfinde nichts."}

                    ]
                    )
    chat_out = json.loads(response.choices[0].message.content)
    output = ""
    for value in chat_out.values():
        output += str(value) 
    return output 

def sanitize_filename(filename):
    """
    Replace non-alphanumeric characters with underscores and ensure a safe filename.
    """
    # Remove directory separators
    filename = os.path.basename(filename)
    # Replace non-alphanumeric characters with underscores
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    return filename

@app.route('/upload', methods=['POST'])
def upload_document():
    """Handles document upload and updates the Documents."""
    if 'file' not in request.files:
        flash('No file part in the request', 'error')
        return redirect(url_for('query_view'))
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('query_view'))

    if file:
        filename = sanitize_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # Ensure the upload directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(file_path)
        # Encode the file path to make it URL-safe
        encoded_path = urllib.parse.quote(file_path)
        
        return redirect(url_for('vectordb', file_path=encoded_path))
      #  return jsonify({"path": file_path}) 
      

@app.route("/", methods=['POST', 'GET']) 
def query_view(): 
    if request.method == 'POST': 
        data = request.get_json()  # Parse JSON payload
        prompt = data.get('prompt', '')  # Get 'prompt' from payload

        response = get_completion(prompt) 
        print(response) 
  
        return jsonify({'response': response}) 
    return render_template('index.html') 

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5801, debug=True) 