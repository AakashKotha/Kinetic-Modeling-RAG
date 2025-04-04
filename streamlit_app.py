# Imports and Initial Setup for RAG Query System

import os
import streamlit as st
import time
import json
import validators
import requests
from io import BytesIO
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
import pymongo
import gridfs
import hashlib

import base64
import numpy as np
import json
import pickle
from datetime import datetime
from io import BytesIO


# Add these imports to your existing imports
import io
import re
import zipfile
import fitz  # PyMuPDF
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
import pikepdf
import openai

# Load environment variables
load_dotenv()

# Predefined admin credentials from environment variables
ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'default_password')

# Session state keys
SESSION_KEYS = [
    "logged_in", 
    "is_admin", 
    "current_page",
    "index", 
    "last_update_time", 
    "index_hash", 
    "indexing_status", 
    "uploaded_files", 
    "urls", 
    "confirm_delete", 
    "confirm_delete_url", 
    "url_input", 
    "data_dir", 
    "should_rerun", 
    "should_clear_url",
    "delete_success_message",
    "delete_error_message", 
    "url_delete_success_message", 
    "url_delete_error_message", 
    "upload_success_message",
    "chat_history", 
    "user_message",    
    "confirm_delete_all"   
]

# Add these constants to your existing constants
GOOGLE_DRIVE_FOLDER_ID = "1Bd4NU6TY_R4MdRHwEXQO8YVmd28EYkfH"  # Your Google Drive folder ID
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Function to hash password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Login page
def login_page():
    st.markdown("<h1 style='text-align: center;'>Kinetic Modeling - RAG</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Access Portal</h2>", unsafe_allow_html=True)
    
    # Create a container to center the columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("Admin Login", key="admin_login_btn", use_container_width=True):
            st.session_state.current_page = 'admin_login'
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("User Access", key="user_login_btn", use_container_width=True):
            st.session_state.logged_in = True
            st.session_state.is_admin = False
            st.session_state.current_page = 'query'
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
# Admin login form
def admin_login_form():
    st.title("Admin Login")
    
    # Use a form to ensure all inputs are captured properly
    with st.form(key="login_form"):
        username = st.text_input("Username", key="admin_username")
        password = st.text_input("Password", type="password", key="admin_password")
        
        # Submit button inside the form
        submit_button = st.form_submit_button("Login")
        
        if submit_button:
            # Add a slight delay to ensure the form values are properly captured
            time.sleep(0.1)
            
            # Compare credentials
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.session_state.logged_in = True
                st.session_state.is_admin = True
                st.session_state.current_page = 'query'
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    # Back button (outside the form)
    if st.button("Back to Access Portal", key="back_to_portal"):
        st.session_state.current_page = 'login'
        st.rerun()

# Core Functions for RAG Query System

# Initialize session state
def initialize_session_state():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.is_admin = False
        st.session_state.current_page = 'login'
        st.session_state.chat_history = [] 
        
        # Initialize other keys from previous implementation
        st.session_state.data_dir = "temp_data"
        if not os.path.exists(st.session_state.data_dir):
            os.makedirs(st.session_state.data_dir)
        
        # OpenAI and MongoDB setup
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            st.error("OpenAI API Key is required. Please set the OPENAI_API_KEY environment variable.")
            return False

        # Set the API key
        os.environ['OPENAI_API_KEY'] = openai_api_key
        
        # MongoDB setup
        MONGO_URI = os.getenv("MONGO_URI")
        if not MONGO_URI:
            st.error("MongoDB URI not found in environment variables. Please check your .env file.")
            return False
        
        try:
            mongo_client = pymongo.MongoClient(
                MONGO_URI, 
                serverSelectionTimeoutMS=30000,  # Increased from 15000
                connectTimeoutMS=30000,          # Increased from 15000
                socketTimeoutMS=60000,           # Increased from 15000
                retryWrites=True,
                tlsAllowInvalidCertificates=True,
                maxPoolSize=100,                 # Increased pool size
                minPoolSize=10,                  # Minimum pool size
                waitQueueTimeoutMS=30000         # Wait queue timeout
            )
            
            # Add more robust connection check
            mongo_client.admin.command('ping', serverSelectionTimeoutMS=10000)
            
            db = mongo_client["rag_system"]
            st.session_state.urls_collection = db["urls"]
            st.session_state.files_collection = db["files"]
            st.session_state.fs = gridfs.GridFS(db)
            
            # Load initial files and URLs
            st.session_state.uploaded_files = [
                file_doc["filename"] for file_doc in st.session_state.files_collection.find({}, {"filename": 1, "_id": 0})
            ]
            st.session_state.urls = [
                url_doc["url"] for url_doc in st.session_state.urls_collection.find({}, {"url": 1, "_id": 0})
            ]
            
        except pymongo.errors.ConnectionFailure as e:
            st.error(f"Failed to connect to MongoDB: {str(e)}")
            st.error("Please check your MongoDB connection settings and network.")
            return False
        except Exception as e:
            st.error(f"Unexpected error connecting to MongoDB: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return False
        
        # Initialize other session state variables
        st.session_state.index = None
        st.session_state.last_update_time = time.time()
        st.session_state.index_hash = ""
        st.session_state.indexing_status = "idle"
        st.session_state.confirm_delete = None
        st.session_state.confirm_delete_url = None
        st.session_state.url_input = ""
        st.session_state.should_rerun = False
        st.session_state.should_clear_url = False
        
        # Message state variables
        st.session_state.delete_success_message = None
        st.session_state.delete_error_message = None
        st.session_state.url_delete_success_message = None
        st.session_state.url_delete_error_message = None
        st.session_state.upload_success_message = None
    
    return True

# Function to get a hash representing current sources (PDFs and URLs)
def get_sources_hash():
    hash_components = []
    
    # Add file metadata to hash
    files = st.session_state.uploaded_files
    for filename in sorted(files):
        file_doc = st.session_state.files_collection.find_one({"filename": filename})
        if file_doc:
            hash_components.append(f"pdf:{filename}:{file_doc.get('last_modified', 0)}")
    
    # Add URLs to hash
    for url in sorted(st.session_state.urls):
        hash_components.append(f"url:{url}")
    
    if not hash_components:
        return "no_sources"
    
    return ";".join(hash_components)

# Function to extract text from URL
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.extract()
        
        text = soup.get_text(separator='\n')
        
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        st.error(f"Error extracting text from URL {url}: {str(e)}")
        return None

# Function to save URLs to MongoDB
def save_urls(urls):
    try:
        # Clear existing URLs and add new ones
        st.session_state.urls_collection.delete_many({})
        for url in urls:
            st.session_state.urls_collection.insert_one({"url": url})
    except Exception as e:
        st.error(f"Error saving URLs to MongoDB: {str(e)}")

# Function to add URL to the knowledge base
def add_url():
    url = st.session_state.url_input.strip()
    
    if not validators.url(url):
        st.session_state.url_delete_error_message = "Please enter a valid URL."
        return False
    
    if url in st.session_state.urls:
        st.session_state.url_delete_error_message = f"URL already exists: {url}"
        return False
    
    st.session_state.urls.append(url)
    save_urls(st.session_state.urls)
    st.session_state.url_delete_success_message = f"Added URL: {url}"
    
    # Don't try to reset url_input directly, use a flag instead
    st.session_state.should_clear_url = True
    st.session_state.index_hash = ""
    st.session_state.should_rerun = True
    return True

# Function to update index with a single document
def update_index_with_document(index, document):
    """Update existing index with a single new document."""
    if index is None:
        return None
        
    try:
        # Convert document to nodes
        nodes = Settings.node_parser.get_nodes_from_documents([document])
        
        # Insert nodes into existing index
        index.insert_nodes(nodes)
        return index
    except Exception as e:
        st.error(f"Error updating index: {str(e)}")
        return index

# Function to handle file upload with better MongoDB connectivity
def handle_file_upload(uploaded_file):
    if uploaded_file is not None:
        try:
            # Create temp directory if it doesn't exist
            if not os.path.exists(st.session_state.data_dir):
                os.makedirs(st.session_state.data_dir)
            
            # First save to temporary file (local storage as backup)
            temp_file_path = os.path.join(st.session_state.data_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                bytes_data = uploaded_file.getbuffer()
                f.write(bytes_data)
            
            # Track the file locally even if MongoDB fails
            if uploaded_file.name not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(uploaded_file.name)
            
            # MongoDB part - try with increased timeout and retry
            max_retries = 3
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    # Create a fresh connection for each retry attempt
                    mongo_client = pymongo.MongoClient(
                                    os.getenv("MONGO_URI"), 
                                    serverSelectionTimeoutMS=30000,
                                    connectTimeoutMS=30000,
                                    socketTimeoutMS=60000,
                                    maxPoolSize=100,
                                    minPoolSize=10,
                                    waitQueueTimeoutMS=30000,
                                    retryWrites=True
                                )
                    
                    db = mongo_client["rag_system"]
                    fs = gridfs.GridFS(db)
                    files_collection = db["files"]
                    
                    # Re-read file from disk to avoid potential buffer issues
                    with open(temp_file_path, "rb") as f:
                        file_content = f.read()
                        
                        # Save to GridFS directly using bytes
                        file_id = fs.put(
                            file_content,
                            filename=uploaded_file.name,
                            content_type=uploaded_file.type,
                            chunkSize=1048576  # Use 1MB chunks to reduce timeouts
                        )
                    
                    # Save file metadata
                    files_collection.insert_one({
                        "filename": uploaded_file.name,
                        "gridfs_id": file_id,
                        "size": len(file_content),
                        "last_modified": time.time()
                    })
                    
                    success = True
                    st.session_state.upload_success_message = f"Uploaded: {uploaded_file.name}"
                    
                    # Try to incrementally update the index if it exists
                    if st.session_state.index is not None:
                        try:
                            reader = SimpleDirectoryReader(input_files=[temp_file_path])
                            new_doc = reader.load_data()[0]
                            st.session_state.index = update_index_with_document(st.session_state.index, new_doc)
                            st.session_state.last_update_time = time.time()
                        except Exception:
                            # Fall back to full reindex if incremental update fails
                            st.session_state.index_hash = ""
                    else:
                        st.session_state.index_hash = ""
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        st.warning(f"MongoDB storage failed after {max_retries} attempts, but file is saved locally and will be used for indexing.")
                        st.session_state.upload_success_message = f"Uploaded locally: {uploaded_file.name} (MongoDB backup failed)"
                        st.session_state.index_hash = ""
                    else:
                        # Wait before retrying
                        time.sleep(1)
                finally:
                    # Close the temporary connection
                    if 'mongo_client' in locals():
                        mongo_client.close()
            
            # Set flag for rerun instead of direct call
            st.session_state.should_rerun = True
            return True
            
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    return False

# Main Application and Indexing Logic for RAG Query System

# Function to load and index documents
def load_and_index_documents():
    try:
        # Create a sentence splitter for more natural chunks
        node_parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50,
            paragraph_separator="\n\n",
            secondary_chunking_regex="(?<=\. )"
        )
        
        # Set the node parser in settings
        Settings.node_parser = node_parser
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50
        
        # Create temp data directory if it doesn't exist
        if not os.path.exists(st.session_state.data_dir):
            os.makedirs(st.session_state.data_dir)
            
        st.session_state.indexing_status = "in_progress"
        
        documents = []
        
        # Ensure we have all files in the temp directory
        for filename in st.session_state.uploaded_files:
            temp_file_path = os.path.join(st.session_state.data_dir, filename)
            
            # If file doesn't exist in temp dir, retrieve from GridFS
            if not os.path.exists(temp_file_path):
                try:
                    file_doc = st.session_state.files_collection.find_one({"filename": filename})
                    if file_doc and "gridfs_id" in file_doc:
                        grid_out = st.session_state.fs.get(file_doc["gridfs_id"])
                        with open(temp_file_path, "wb") as f:
                            f.write(grid_out.read())
                except Exception as e:
                    st.warning(f"Error retrieving {filename}: {str(e)}")
        
        # Load PDF documents from temp directory
        pdf_files = [f for f in os.listdir(st.session_state.data_dir) if f.endswith(".pdf")]
        if pdf_files:
            try:
                reader = SimpleDirectoryReader(st.session_state.data_dir)
                pdf_documents = reader.load_data()
                documents.extend(pdf_documents)
            except Exception as e:
                st.error(f"Error reading PDF documents: {str(e)}")
        
        # Load URL content
        for url in st.session_state.urls:
            try:
                text = extract_text_from_url(url)
                if text:
                    doc = Document(text=text, metadata={"source": url, "type": "url"})
                    documents.append(doc)
            except Exception as e:
                st.warning(f"Error processing URL {url}: {str(e)}")
        
        if not documents:
            st.session_state.indexing_status = "idle"
            return None
            
        # Create index from documents
        index = VectorStoreIndex.from_documents(documents)
        
        st.session_state.indexing_status = "complete"
        return index
    except Exception as e:
        st.session_state.indexing_status = "idle"
        st.error(f"Error loading or indexing documents: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Function to create optimized query engine
def create_optimized_query_engine(index):
    # Increase top_k for better coverage
    retriever = VectorIndexRetriever(index=index, similarity_top_k=6)
    
    # Add a relevance filter to improve results
    node_postprocessors = [SimilarityPostprocessor(similarity_cutoff=0.7)]
    
    return RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=node_postprocessors
    )

# Function to set the file to be deleted with confirmation
def set_delete_confirmation(filename):
    # Clear any previous confirmation first
    st.session_state.confirm_delete_url = None
    # Set the current confirmation
    st.session_state.confirm_delete = filename
    # Force a direct rerun since this is triggered by a button click
    st.rerun()

# Function to set the URL to be deleted with confirmation
def set_delete_url_confirmation(url):
    # Clear any previous confirmation first
    st.session_state.confirm_delete = None
    # Set the current confirmation
    st.session_state.confirm_delete_url = url
    # Force a direct rerun since this is triggered by a button click
    st.rerun()

def confirm_delete():
    if st.session_state.confirm_delete:
        filename = st.session_state.confirm_delete
        
        try:
            # Remove from GridFS first
            file_doc = st.session_state.files_collection.find_one({"filename": filename})
            if file_doc and "gridfs_id" in file_doc:
                try:
                    grid_id = file_doc["gridfs_id"]
                    st.session_state.fs.delete(grid_id)
                except Exception as e:
                    st.error(f"Error deleting from GridFS: {str(e)}")
            
            # Then remove from metadata collection
            st.session_state.files_collection.delete_one({"filename": filename})
            
            # Then remove from temp directory
            temp_file_path = os.path.join(st.session_state.data_dir, filename)
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            # Force complete reindex
            st.session_state.index_hash = ""
            
            # Finally remove from session state list
            if filename in st.session_state.uploaded_files:
                st.session_state.uploaded_files.remove(filename)
            
            # Refresh MongoDB list to be certain
            st.session_state.uploaded_files = [
                file_doc["filename"] for file_doc in st.session_state.files_collection.find({}, {"filename": 1, "_id": 0})
            ]
            
            # Store a success message
            st.session_state.delete_success_message = f"Deleted: {filename}"
            
        except Exception as e:
            st.session_state.delete_error_message = f"Error deleting file: {str(e)}"
            import traceback
            st.error(traceback.format_exc())
            
        # Clear the confirmation
        st.session_state.confirm_delete = None
        # Force direct rerun
        st.rerun()

# Function to confirm deletion of URL
def confirm_delete_url():
    if st.session_state.confirm_delete_url:
        url = st.session_state.confirm_delete_url
        
        try:
            # Check if URL exists in current list
            if url in st.session_state.urls:
                st.session_state.urls.remove(url)
                # Save updated URLs to MongoDB
                save_urls(st.session_state.urls)
                st.session_state.index_hash = ""
                st.session_state.url_delete_success_message = f"Removed URL: {url}"
            else:
                st.session_state.url_delete_error_message = f"URL not found: {url}"
        except Exception as e:
            st.session_state.url_delete_error_message = f"Error removing URL: {str(e)}"
            import traceback
            st.error(traceback.format_exc())
            
        # Clear the confirmation
        st.session_state.confirm_delete_url = None
        # Force direct rerun since this is triggered by a button click
        st.rerun()

# Function to cancel deletion
def cancel_delete():
    st.session_state.confirm_delete = None
    st.session_state.confirm_delete_url = None
    # Force direct rerun since this is triggered by a button click
    st.rerun()

def generate_suggested_questions(query_engine, force_refresh=False):
    """Generate suggested questions based on the current content in the knowledge base."""
    try:
        # System prompt to generate questions based on current knowledge base content
        system_prompt = """
        Based on the current content in the knowledge base, generate 3 diverse and interesting starter questions 
        that users might want to ask. The questions should:
        
        1. Be directly answerable from the CURRENT knowledge base content
        2. Cover different topics or aspects of the available documents
        3. Be concise (10 words or less) but specific enough to be meaningful
        4. Reflect the most recent and relevant information in the documents
        5. Be in the form of questions (end with ?)
        
        IMPORTANT: Only generate questions based on the CURRENT documents available.
        
        Format the response as a numbered list with just the questions, no additional text.
        """

        # Use the query engine to generate suggestions
        response = query_engine.query(system_prompt)
        
        # Parse the response to extract the questions
        suggested_questions = []
        for line in response.response.strip().split('\n'):
            # Look for numbered lines (1., 2., 3., etc.)
            if line.strip() and (line.strip()[0].isdigit() and line.strip()[1:3] in ['. ', '? ', ') ']):
                # Extract just the question, removing the number and any trailing punctuation
                question = line.strip()[3:].strip()
                if question:
                    suggested_questions.append(question)
        
        # If we didn't get exactly 3 questions, provide fallbacks based on current content
        if len(suggested_questions) != 3:
            suggested_questions = [
                "What documents are currently in the knowledge base?",
                "What topics are covered in the available sources?",
                "Can you summarize the main content of current documents?"
            ]
        
        return suggested_questions
    
    except Exception:
        # If there's any error, use generic suggestions that prompt exploration
        return [
            "What documents are currently available?",
            "What kind of information can I find here?",
            "Can you help me understand the current knowledge base?"
        ]

def display_chat_interface(query_engine):
    # Display chat history
    for i, (user_msg, assistant_msg) in enumerate(st.session_state.chat_history):
        # User message
        with st.chat_message("user"):
            st.write(user_msg)
        # Assistant message
        with st.chat_message("assistant"):
            st.write(assistant_msg)
    
    # Show suggested questions only if this is the first interaction (empty chat history)
    if len(st.session_state.chat_history) == 0 and query_engine is not None:
        st.write("Suggested questions:")
        
        # Generate suggested questions based on knowledge base
        try:
            suggested_questions = generate_suggested_questions(query_engine)
        except Exception:
            # Fallback if generation fails
            suggested_questions = [
                "What documents are currently available?",
                "What topics can I explore?",
                "Can you help me understand the knowledge base?"
            ]
        
        # Display clickable buttons for each suggested question
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(suggested_questions[0], key="suggested_q1"):
                process_new_message(suggested_questions[0], query_engine)
                st.rerun()
                
        with col2:
            if st.button(suggested_questions[1], key="suggested_q2"):
                process_new_message(suggested_questions[1], query_engine)
                st.rerun()
                
        with col3:
            if st.button(suggested_questions[2], key="suggested_q3"):
                process_new_message(suggested_questions[2], query_engine)
                st.rerun()
    
    # Use a container for better alignment control
    with st.container():
        # Create a layout with carefully adjusted widths
        col1, col2, col3 = st.columns([0.07, 0.78, 0.15])
        
        # Add CSS for vertical alignment and text area height
        st.markdown("""
        <style>
        .stColumn > div {
            display: flex;
            align-items: center;
            height: 100%;
        }
        /* Custom styling for the text area to make it 3 lines tall with scrolling */
        textarea {
            min-height: 80px !important;  /* Approximately 3 lines */
            max-height: 80px !important;
            overflow-y: auto !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Place the icon in the first column
        with col1:
            # Use chat_message for consistent icon styling
            with st.chat_message("user"):
                st.write("")
        
        # Place the text area in the second column
        with col2:
            # Get the appropriate placeholder
            placeholder = "Enter your query" if len(st.session_state.chat_history) == 0 else "Enter your follow-up question"
            
            user_message = st.text_area("", 
                                        key="user_message", 
                                        placeholder=placeholder,
                                        label_visibility="collapsed")
        
        # Place the send button in the third column 
        with col3:
            send_pressed = st.button("Send", key="send_message", use_container_width=True)
    
    # Process message
    if send_pressed and user_message:
        process_new_message(user_message, query_engine)
        st.rerun()

def process_new_message(user_message, query_engine):
    if not user_message:
        return

    if query_engine is None:
        assistant_response = "I'm sorry, but the knowledge base isn't available. Please try again later."
    else:
        try:
            # Build conversation context with recency bias
            context = ""
            if st.session_state.chat_history:
                context += "Here is the full conversation history so far:\n\n"
                for i, (u, a) in enumerate(st.session_state.chat_history):
                    if i == len(st.session_state.chat_history) - 1:
                        context += f"[Most recent exchange]\nUser: {u}\nAssistant: {a}\n\n"
                    else:
                        context += f"User: {u}\nAssistant: {a}\n\n"

            # Refined and clear prompt
            full_query = f"""
            You are an intelligent AI assistant helping a user in an ongoing conversation.

            USER'S CURRENT MESSAGE:
            \"{user_message}\"

            CONTEXT:
            {context}

            INSTRUCTIONS:
            - Give more proiority to the "Most recent exchange", if the query is a follow-up. 
            - If the user's message is a follow-up, continue the topic accordingly using the most recent exchange.
            - If it's a new or unrelated question, find relevant context in the conversation history.
            - If clarification is needed, ask a concise follow-up question.
            - Prioritize clarity, relevance, and helpfulness.
            - Keep your tone friendly and informative.

            Respond to the user's message with a thoughtful, concise, and helpful reply.
            """

            # Query the engine
            response = query_engine.query(full_query)
            assistant_response = response.response if response.response else "I couldn't find a relevant answer to your question."

        except Exception as e:
            assistant_response = f"I encountered an error: {str(e)}"
            import traceback
            st.error(traceback.format_exc())

    # Save new exchange
    st.session_state.chat_history.append((user_message, assistant_response))

# Function to export embeddings in a suitable format for comparative analysis
def export_embeddings(index):
    """
    Exports the embeddings from the VectorStoreIndex in a format optimized for 
    manifold comparison and visualization.
    
    Args:
        index: The VectorStoreIndex instance containing embeddings
        
    Returns:
        A tuple of (success, file_content or error_message)
    """
    try:
        if index is None:
            return False, "No index available. Please add documents and index them first."
        
        # Initialize data structure to store embeddings and metadata
        embeddings_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "doc_count": len(st.session_state.uploaded_files),
                "url_count": len(st.session_state.urls),
                "description": "Embeddings export for manifold comparison"
            },
            "nodes": []
        }
        
        # Get all nodes from the index using different approaches based on API version
        nodes = []
        
        # Try different ways to access nodes from the docstore
        try:
            # Approach 1: Check if we can directly get all documents 
            if hasattr(index, "docstore") and hasattr(index.docstore, "get_all_documents"):
                node_dict = index.docstore.get_all_documents()
                nodes = list(node_dict.values())
            # Approach 2: Access _docstore._nodes if available
            elif hasattr(index, "_docstore") and hasattr(index._docstore, "_nodes"):
                node_dict = index._docstore._nodes
                nodes = list(node_dict.values())
            # Approach 3: Access index.index_struct.nodes for older versions
            elif hasattr(index, "index_struct") and hasattr(index.index_struct, "nodes"):
                nodes = index.index_struct.nodes
            # Approach 4: Try to get documents through storage context
            elif hasattr(index, "_storage_context") and hasattr(index._storage_context, "docstore"):
                if hasattr(index._storage_context.docstore, "docs"):
                    node_dict = index._storage_context.docstore.docs
                    nodes = list(node_dict.values())
                elif hasattr(index._storage_context.docstore, "get_all_documents"):
                    node_dict = index._storage_context.docstore.get_all_documents()
                    nodes = list(node_dict.values())
            # If no nodes found, try to get nodes directly from the query engine's retriever
            elif hasattr(index, "as_query_engine"):
                query_engine = index.as_query_engine()
                if hasattr(query_engine, "_retriever") and hasattr(query_engine._retriever, "_nodes"):
                    nodes = query_engine._retriever._nodes
        except Exception as e:
            st.warning(f"Error accessing nodes using standard methods: {str(e)}")
            
        # If we still have no nodes, try to retrieve from all index nodes in a different way
        if not nodes:
            try:
                # Try to use the index's get_all_nodes method
                if hasattr(index, "get_all_nodes"):
                    nodes = index.get_all_nodes()
                # Try to use a dummy query to get some nodes
                elif hasattr(index, "as_query_engine"):
                    query_engine = index.as_query_engine()
                    response = query_engine.query("summarize all the content")
                    if hasattr(response, "source_nodes"):
                        nodes = [node.node for node in response.source_nodes]
            except Exception as e:
                st.warning(f"Error in fallback node retrieval: {str(e)}")
                
        # If we still have no nodes, return an error
        if not nodes:
            return False, "Could not access the nodes in the index. This may be due to API changes in llama_index."
        
        # Try to get the embed model from the index
        embed_model = None
        try:
            if hasattr(index, "_embed_model"):
                embed_model = index._embed_model
            elif hasattr(index, "embeddings") and hasattr(index.embeddings, "embed_model"):
                embed_model = index.embeddings.embed_model
            elif hasattr(index, "_service_context") and hasattr(index._service_context, "embed_model"):
                embed_model = index._service_context.embed_model
        except Exception:
            embed_model = None
        
        # Process each node and try to get its embedding
        for i, node in enumerate(nodes):
            try:
                node_id = str(getattr(node, "id", f"node_{i}"))
                
                # Get text content
                if hasattr(node, "get_content"):
                    text = node.get_content()
                elif hasattr(node, "text"):
                    text = node.text
                elif hasattr(node, "content"):
                    text = node.content
                else:
                    text = str(node)
                
                # Get metadata
                if hasattr(node, "metadata"):
                    metadata = node.metadata
                elif hasattr(node, "extra_info"):
                    metadata = node.extra_info
                else:
                    metadata = {}
                
                # Try to get embedding - different approaches
                embedding = None
                
                # 1. Try to get embedding directly from node if it has one
                if hasattr(node, "embedding"):
                    embedding = node.embedding
                
                # 2. Try to get from vector store if possible
                if embedding is None and embed_model is not None:
                    try:
                        embedding = embed_model.get_text_embedding(text)
                    except:
                        pass
                
                # 3. If we still don't have an embedding, use a fallback approach with OpenAI directly
                if embedding is None and "OPENAI_API_KEY" in os.environ:
                    try:
                        import openai
                        openai.api_key = os.environ["OPENAI_API_KEY"]
                        embedding_response = openai.Embedding.create(
                            model="text-embedding-ada-002",
                            input=text[:8191]  # Truncate to max token limit
                        )
                        embedding = embedding_response["data"][0]["embedding"]
                    except:
                        # If OpenAI direct approach fails, just use a placeholder
                        # This allows for exporting node text even without embeddings
                        embedding = [0.0] * 768  # Standard embedding size
                
                # Convert numpy arrays to lists for JSON serialization
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                
                # Add node information
                node_data = {
                    "id": node_id,
                    "text": text,
                    "metadata": metadata,
                    "embedding": embedding
                }
                
                embeddings_data["nodes"].append(node_data)
                
            except Exception as e:
                # Skip problematic nodes but continue processing others
                st.warning(f"Could not process node {i}: {str(e)}")
                continue
        
        # Check if we have any nodes processed
        if not embeddings_data["nodes"]:
            return False, "No embeddings could be exported. Please try adding more documents to your index."
        
        # Generate a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Provide two format options
        
        # Option 1: JSON (human-readable, interoperable)
        json_content = json.dumps(embeddings_data, indent=2)
        json_filename = f"embeddings_export_{timestamp}.json"
        
        # Option 2: Pickle (efficient for large numpy arrays, Python-specific)
        pickle_buffer = BytesIO()
        pickle.dump(embeddings_data, pickle_buffer)
        pickle_content = pickle_buffer.getvalue()
        pickle_filename = f"embeddings_export_{timestamp}.pkl"
        
        return True, {
            "json": (json_content, json_filename),
            "pickle": (pickle_content, pickle_filename)
        }
    
    except Exception as e:
        import traceback
        error_msg = f"Error exporting embeddings: {str(e)}\n{traceback.format_exc()}"
        return False, error_msg

# Function to create a download link for a text file (e.g., JSON)
def get_text_download_link(file_content, file_name, display_text):
    """
    Generates a download link for a text file.
    
    Args:
        file_content: Content of the file to be downloaded
        file_name: Name of the file
        display_text: Text to display for the download link
        
    Returns:
        HTML string with the download link
    """
    b64 = base64.b64encode(file_content.encode()).decode()
    href = f'<a href="data:text/plain;base64,{b64}" download="{file_name}">{display_text}</a>'
    return href

# Function to create a download link for a binary file (e.g., Pickle)
def get_binary_download_link(file_content, file_name, display_text):
    """
    Generates a download link for a binary file.
    
    Args:
        file_content: Binary content of the file to be downloaded
        file_name: Name of the file
        display_text: Text to display for the download link
        
    Returns:
        HTML string with the download link
    """
    b64 = base64.b64encode(file_content).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">{display_text}</a>'
    return href

# Add this function to your code
def delete_all_files():
    """
    Delete all PDF files from MongoDB GridFS, the files collection, and the local temp directory.
    Returns success status and message.
    """
    try:
        # Get all files from the collection
        all_files = list(st.session_state.files_collection.find({}, {"filename": 1, "gridfs_id": 1, "_id": 0}))
        
        # Count the total files
        total_files = len(all_files)
        
        if total_files == 0:
            return True, "No files to delete."
        
        # First delete all files from GridFS
        deleted_count = 0
        for file_doc in all_files:
            try:
                if "gridfs_id" in file_doc:
                    # Delete from GridFS
                    st.session_state.fs.delete(file_doc["gridfs_id"])
                
                # Delete from local temp directory
                if "filename" in file_doc:
                    temp_file_path = os.path.join(st.session_state.data_dir, file_doc["filename"])
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting file {file_doc.get('filename', 'unknown')}: {str(e)}")
                # Continue with next file even if one fails
                continue
        
        # Then delete all entries from the files collection
        st.session_state.files_collection.delete_many({})
        
        # Clear the uploaded_files list in session state
        st.session_state.uploaded_files = []
        
        # Force reindex
        st.session_state.index_hash = ""
        
        return True, f"Successfully deleted {deleted_count} out of {total_files} files."
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error deleting all files: {str(e)}\n{error_details}")
        return False, f"Error deleting files: {str(e)}"

# Add this to your main function in the PDF Documents tab section
# Place it after the "Available PDFs:" heading and before displaying the PDF list

# Add a "Delete All Files" button with confirmation dialog
def add_delete_all_button_to_tab1():
    # Add a delete all button with a red style
    st.markdown(
        """
        <style>
        .delete-all-btn {
            background-color: #FF5252;
            color: white;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    # Use a container to place the button on the right
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Delete All Files", key="delete_all_btn", type="primary", 
                        help="Delete all PDF files from the knowledge base"):
                # Set confirmation state
                st.session_state.confirm_delete_all = True
                st.rerun()
    
    # Show confirmation dialog if needed
    if hasattr(st.session_state, 'confirm_delete_all') and st.session_state.confirm_delete_all:
        st.warning("⚠️ Are you sure you want to delete ALL files? This action cannot be undone.")
        col1, col2 = st.columns(2)
        if col1.button("Yes, Delete All", key="confirm_delete_all_yes", type="primary"):
            # Perform the deletion
            success, message = delete_all_files()
            if success:
                st.session_state.delete_success_message = message
            else:
                st.session_state.delete_error_message = message
            
            # Clear confirmation state
            st.session_state.confirm_delete_all = False
            st.rerun()
            
        if col2.button("Cancel", key="confirm_delete_all_no"):
            # Clear confirmation state
            st.session_state.confirm_delete_all = False
            st.rerun()

# Function to authenticate with Google Drive
def authenticate_google_drive():
    """Authenticate with Google Drive API."""
    creds = None
    # The file token.json stores the user's access and refresh tokens
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # If credentials don't exist or are invalid, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    return creds

# Function to list PDF files in the specified Google Drive folder
def list_pdf_files_in_drive(service, folder_id):
    """List all PDF files in the specified Google Drive folder."""
    results = []
    page_token = None
    
    while True:
        response = service.files().list(
            q=f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false",
            spaces='drive',
            fields='nextPageToken, files(id, name, size, modifiedTime)',
            pageToken=page_token
        ).execute()
        
        results.extend(response.get('files', []))
        page_token = response.get('nextPageToken')
        
        if not page_token:
            break
    
    return results

def extract_text_for_analysis(pdf_content, max_pages=10):
    """
    Extract text from PDF with improved robustness and logging.
    
    Args:
        pdf_content (bytes): PDF file content
        max_pages (int): Maximum number of pages to extract
    
    Returns:
        str: Extracted text
    """
    extracted_text = ""
    try:
        with io.BytesIO(pdf_content) as pdf_buffer:
            doc = fitz.open(stream=pdf_buffer, filetype="pdf")
            
            # Try multiple approaches to extract text
            for page_num in range(min(max_pages, doc.page_count)):
                try:
                    page = doc[page_num]
                    
                    # Extract text with multiple methods
                    page_text = page.get_text("text")  # Standard text extraction
                    
                    # If standard extraction fails, try alternative methods
                    if not page_text.strip():
                        # Try extracting blocks
                        blocks = page.get_text("blocks")
                        page_text = " ".join([block[4] for block in blocks if block[4].strip()])
                    
                    # Add page number for context
                    extracted_text += f"--- Page {page_num + 1} ---\n{page_text}\n\n"
                
                except Exception as page_error:
                    print(f"Error extracting text from page {page_num}: {page_error}")
            
            # Limit total text length
            if len(extracted_text) > 15000:
                extracted_text = extracted_text[:15000] + "... [TEXT TRUNCATED]"
            
            # Debug logging
            print("Extracted PDF Text Length:", len(extracted_text))
            print("First 500 characters:\n", extracted_text[:500])
            
            return extracted_text
    
    except Exception as e:
        print(f"Critical error in text extraction: {e}")
        import traceback
        traceback.print_exc()
        return ""

def parse_original_filename(filename):
    """
    Parse the original filename to extract potential metadata.
    
    Args:
        filename (str): Original filename of the PDF
    
    Returns:
        dict: Parsed filename metadata
    """
    # Remove file extension
    basename = os.path.splitext(filename)[0]
    
    # Default values
    parsed = {
        "year": "Unknown",
        "author": "Unknown",
        "title": clean_title(basename),
        "category": "Advanced"  # Default category
    }
    
    # Try to extract year (looking for 4 digits that could be a year)
    year_match = re.search(r'(19|20)\d{2}', basename)
    if year_match:
        parsed["year"] = year_match.group(0)
    
    # Try to extract author - multiple patterns
    author_patterns = [
        r'([A-Z][a-z]+),\s*[A-Z]',  # LastName, FirstInitial
        r'([A-Z][a-z]+)\s+et\s+al',  # LastName et al
        r'([A-Z][a-z]+)\s+[A-Z]\.',  # LastName F.
    ]
    
    for pattern in author_patterns:
        author_match = re.search(pattern, basename)
        if author_match:
            parsed["author"] = author_match.group(1)
            break
    
    return parsed

def clean_title(title):
    """
    Clean and format a title string for filename.
    
    Args:
        title (str): Original title
    
    Returns:
        str: Cleaned and formatted title
    """
    # Remove special characters
    title = re.sub(r'[<>:"/\\|?*]', '', title)
    
    # Replace spaces and underscores with hyphens
    title = re.sub(r'[\s_]+', '-', title)
    
    # Remove small/common words
    words = title.split('-')
    significant_words = [
        w for w in words 
        if len(w) > 2 and 
        w.lower() not in ["the", "and", "for", "with", "this", "that", "from", "using"]
    ]
    
    # Take up to 6 significant words
    title = '-'.join(significant_words[:6])
    
    return title

def clean_filename(filename):
    """
    Clean up filename to ensure it meets requirements.
    
    Args:
        filename (str): Original filename
    
    Returns:
        str: Cleaned filename
    """
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    
    # Replace spaces with hyphens
    filename = re.sub(r'\s+', '-', filename)
    
    # Remove consecutive hyphens
    filename = re.sub(r'-+', '-', filename)
    
    # Ensure filename isn't too long
    if len(filename) > 255:
        parts = filename[:-4].split('_')
        if len(parts) >= 4:
            # Shorten the title part
            max_title_length = 100 - len(f"{parts[0]}_{parts[1]}_{parts[3]}.pdf") - 3
            parts[2] = parts[2][:max_title_length]
            filename = '_'.join(parts) + '.pdf'
    
    return filename

def create_standardized_filename(parsed_info):
    """
    Create a basic standardized filename from parsed information.
    
    Args:
        parsed_info (dict): Parsed filename metadata
    
    Returns:
        str: Standardized filename
    """
    return f"{parsed_info['year']}_{parsed_info['author']}_{parsed_info['title']}_{parsed_info.get('category', 'Advanced')}.pdf"

def get_standardized_filename_with_chatgpt(original_filename, pdf_content):
    """
    Use ChatGPT to generate a standardized, meaningful filename for academic papers.
    
    Args:
        original_filename (str): Original filename of the PDF
        pdf_content (bytes): Content of the PDF file
    
    Returns:
        str: Standardized filename
    """
    # Parse original filename for fallback information
    parsed_info = parse_original_filename(original_filename)
    
    try:
        # Extract text from the PDF for analysis
        text = extract_text_for_analysis(pdf_content, max_pages=10)
        
        # If no text extracted, use a different strategy
        if not text.strip():
            print(f"WARNING: No text extracted from {original_filename}")
            return create_standardized_filename(parsed_info)
        
        # Create OpenAI client
        client = openai.OpenAI()
        
        # Attempt to use GPT-4 if available (more accurate)
        try:
            completion = client.chat.completions.create(
                model="gpt-4-1106-preview",  # Use GPT-4 Turbo
                messages=[
                    {
                        "role": "system", 
                        "content": """
                        You are an expert academic document classifier for scientific papers.
                        Extract precise metadata from the document:
                        
                        1. Year: Most recent/prominent publication year (4 digits)
                        2. Author: FIRST AUTHOR'S LAST NAME ONLY
                           - If multiple authors, use the first author's last name
                           - No initials or full names
                        3. Title: 5-6 MOST TECHNICAL words from the paper title
                           - Remove common words (the, a, an, etc.)
                           - Use technical, specific terminology
                           - Hyphen-separate words
                        4. Category: Most appropriate from:
                           Brain, Lung, Liver, Heart, Kidney, Advanced, Other
                        
                        Respond EXACTLY in this JSON format:
                        {
                          "year": "YYYY",
                          "author": "LastName",
                          "title": "Technical-Words-From-Title",
                          "category": "Category"
                        }
                        """
                    },
                    {
                        "role": "user", 
                        "content": f"""
                        Original Filename: {original_filename}
                        
                        Document Text:
                        {text[:15000]}  # Limit to avoid token overflow
                        
                        Please carefully analyze the document and extract the metadata.
                        """
                    }
                ],
                temperature=0.3,
                max_tokens=250,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            response_text = completion.choices[0].message.content.strip()
            metadata = json.loads(response_text)
            
            # Validate and clean the metadata
            year = metadata.get('year', parsed_info['year'])
            author = metadata.get('author', parsed_info['author'])
            title = metadata.get('title', parsed_info['title'])
            category = metadata.get('category', 'Advanced')
            
            # Create and clean the filename
            filename = f"{year}_{author}_{title}_{category}.pdf"
            return clean_filename(filename)
        
        except Exception as gpt4_error:
            print(f"GPT-4 error: {gpt4_error}")
            # Fallback to GPT-3.5 if GPT-4 fails
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {
                        "role": "system", 
                        "content": """
                        Extract document metadata precisely:
                        1. Year: 4-digit publication year
                        2. Author: First author's LAST NAME only
                        3. Title: 5-6 technical words (hyphen-separated)
                        4. Category: Brain, Lung, Liver, Heart, Kidney, Advanced, Other
                        
                        JSON format required:
                        {
                          "year": "YYYY",
                          "author": "LastName",
                          "title": "Technical-Words-From-Title",
                          "category": "Category"
                        }
                        """
                    },
                    {
                        "role": "user", 
                        "content": f"""
                        Original Filename: {original_filename}
                        
                        Document Text:
                        {text[:15000]}
                        
                        Extract metadata carefully.
                        """
                    }
                ],
                temperature=0.4,
                max_tokens=250,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            response_text = completion.choices[0].message.content.strip()
            metadata = json.loads(response_text)
            
            # Validate and clean the metadata
            year = metadata.get('year', parsed_info['year'])
            author = metadata.get('author', parsed_info['author'])
            title = metadata.get('title', parsed_info['title'])
            category = metadata.get('category', 'Advanced')
            
            # Create and clean the filename
            filename = f"{year}_{author}_{title}_{category}.pdf"
            return clean_filename(filename)
    
    except Exception as e:
        print(f"Error generating standardized filename: {e}")
        import traceback
        traceback.print_exc()
        
        # Final fallback to create a basic standardized name
        return create_standardized_filename(parsed_info)

def compress_pdf(pdf_content):
    """
    Compress PDF content using pikepdf with maximum compression settings.
    
    Args:
        pdf_content (bytes): Original PDF file content
    
    Returns:
        bytes: Compressed PDF content
    """
    try:
        # Create a temporary directory for processing
        temp_dir = "pdf_compression_temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate unique temporary filenames
        import uuid
        input_filename = os.path.join(temp_dir, f"input_{uuid.uuid4()}.pdf")
        output_filename = os.path.join(temp_dir, f"compressed_{uuid.uuid4()}.pdf")
        
        # Write input PDF to temporary file
        with open(input_filename, "wb") as f:
            f.write(pdf_content)
        
        # Open the PDF
        pdf = pikepdf.open(input_filename)
        
        # Aggressive compression settings
        pdf.save(
            output_filename, 
            compress_streams=True,          # Compress stream objects
            compress_images=True,           # Compress images aggressively
            object_stream_mode=pikepdf.ObjectStreamMode.generate,  # Optimize object streams
            linearize=True,                 # Create linearized (web-optimized) PDF
            remove_unreferenced_resources=True,  # Remove unused resources
            
            # Advanced compression parameters
            stream_data_mode=pikepdf.StreamDataMode.compress,  # Maximize stream compression
            optimize_mode=pikepdf.OptimizeMode.full,  # Full optimization
        )
        
        # Read compressed content
        with open(output_filename, "rb") as f:
            compressed_content = f.read()
        
        # Calculate compression ratio
        original_size = len(pdf_content)
        compressed_size = len(compressed_content)
        compression_ratio = (original_size - compressed_size) / original_size * 100
        
        # Clean up temporary files
        try:
            os.remove(input_filename)
            os.remove(output_filename)
            os.rmdir(temp_dir)
        except Exception as cleanup_error:
            print(f"Warning: Could not remove temporary files: {cleanup_error}")
        
        # Only return compressed version if it's smaller
        if compressed_size < original_size:
            print(f"PDF Compression: {compression_ratio:.2f}% reduction")
            return compressed_content
        
        # If no significant compression, return original content
        print("PDF compression did not reduce file size")
        return pdf_content
    
    except Exception as e:
        print(f"Error during PDF compression: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # Return original content if compression fails
        return pdf_content

# Function to download and process a PDF file from Google Drive
def download_and_process_pdf(service, file_id, original_filename):
    """
    Download a PDF file from Google Drive, use ChatGPT to standardize the filename,
    compress it, and return it.
    """
    try:
        # Create a BytesIO object to store the downloaded file
        file_buffer = io.BytesIO()
        
        # Create a MediaIoBaseDownload object for the file
        request = service.files().get_media(fileId=file_id)
        downloader = MediaIoBaseDownload(file_buffer, request)
        
        # Download the file
        done = False
        while not done:
            status, done = downloader.next_chunk()
        
        # Get the file content
        file_content = file_buffer.getvalue()
        
        # Use ChatGPT to get a standardized filename
        standardized_filename = get_standardized_filename_with_chatgpt(
            original_filename, 
            file_content
        )
        
        # Compress the PDF
        compressed_content = compress_pdf(file_content)
        
        # Calculate compression ratio
        original_size = len(file_content)
        compressed_size = len(compressed_content)
        compression_ratio = (original_size - compressed_size) / original_size * 100
        
        return {
            'original_filename': original_filename,
            'standardized_filename': standardized_filename,
            'content': compressed_content,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio
        }
    except Exception as e:
        print(f"Error downloading/processing file {original_filename}: {str(e)}")
        return None

# Function to import PDFs from Google Drive and save to MongoDB
def import_pdfs_from_google_drive():
    """
    Main function to import PDFs from Google Drive, process them using ChatGPT,
    and save to MongoDB. Includes rate limiting for ChatGPT API calls.
    """
    results = {
        'success': False,
        'total_files': 0,
        'processed_files': 0,
        'error_files': 0,
        'total_size_original': 0,
        'total_size_compressed': 0,
        'files': []
    }
    
    mongo_client = None
    try:
        # Authenticate with Google Drive
        creds = authenticate_google_drive()
        service = build('drive', 'v3', credentials=creds)
        
        # List PDF files in the specified folder
        pdf_files = list_pdf_files_in_drive(service, GOOGLE_DRIVE_FOLDER_ID)
        results['total_files'] = len(pdf_files)
        
        if not pdf_files:
            results['message'] = "No PDF files found in the specified Google Drive folder."
            return results
        
        # Set OpenAI API key from environment variable
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Create MongoDB client with robust connection settings
        mongo_client = pymongo.MongoClient(
            os.getenv("MONGO_URI"), 
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=30000,
            socketTimeoutMS=60000,
            maxPoolSize=100,
            minPoolSize=10,
            waitQueueTimeoutMS=30000,
            retryWrites=True,
            tlsAllowInvalidCertificates=True
        )
        
        # Verify MongoDB connection
        mongo_client.admin.command('ping', serverSelectionTimeoutMS=10000)
        
        # Get database and collections
        db = mongo_client["rag_system"]
        files_collection = db["files"]
        fs = gridfs.GridFS(db)
        
        # Add a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each PDF file with rate limiting
        for i, pdf_file in enumerate(pdf_files):
            file_id = pdf_file['id']
            original_filename = pdf_file['name']
            
            # Update progress
            progress_bar.progress((i + 1) / len(pdf_files))
            status_text.text(f"Processing file {i+1} of {len(pdf_files)}: {original_filename}")
            
            # Check if this file already exists in the database
            if files_collection.find_one({"original_filename": original_filename}):
                # Skip this file and count it as processed
                results['processed_files'] += 1
                results['files'].append({
                    'original_filename': original_filename,
                    'status': 'skipped',
                    'message': 'File already exists in the database'
                })
                continue
            
            # Download and process the PDF with ChatGPT
            processed_file = download_and_process_pdf(service, file_id, original_filename)
            
            if processed_file:
                try:
                    # Save to GridFS
                    gridfs_file_id = fs.put(
                        processed_file['content'],
                        filename=processed_file['standardized_filename'],
                        content_type='application/pdf',
                        original_filename=original_filename
                    )
                    
                    # Save file metadata to MongoDB
                    files_collection.insert_one({
                        "filename": processed_file['standardized_filename'],
                        "original_filename": original_filename,
                        "gridfs_id": gridfs_file_id,
                        "size": processed_file['compressed_size'],
                        "original_size": processed_file['original_size'],
                        "compression_ratio": processed_file['compression_ratio'],
                        "source": "google_drive",
                        "drive_file_id": pdf_file['id'],
                        "last_modified": time.time()
                    })
                    
                    # Save to local temp directory
                    temp_file_path = os.path.join(st.session_state.data_dir, processed_file['standardized_filename'])
                    with open(temp_file_path, "wb") as f:
                        f.write(processed_file['content'])
                    
                    # Add to session state uploaded files
                    if processed_file['standardized_filename'] not in st.session_state.uploaded_files:
                        st.session_state.uploaded_files.append(processed_file['standardized_filename'])
                    
                    # Update results statistics
                    results['processed_files'] += 1
                    results['total_size_original'] += processed_file['original_size']
                    results['total_size_compressed'] += processed_file['compressed_size']
                    results['files'].append({
                        'original_filename': original_filename,
                        'standardized_filename': processed_file['standardized_filename'],
                        'status': 'success',
                        'compression_ratio': processed_file['compression_ratio']
                    })
                except Exception as insert_error:
                    # Record error during MongoDB insertion
                    results['error_files'] += 1
                    results['files'].append({
                        'original_filename': original_filename,
                        'status': 'error',
                        'message': f'Failed to insert file: {str(insert_error)}'
                    })
            else:
                # Record error during PDF processing
                results['error_files'] += 1
                results['files'].append({
                    'original_filename': original_filename,
                    'status': 'error',
                    'message': 'Failed to process file'
                })
            
            # Add a delay between requests to avoid rate limits (mainly for OpenAI API)
            if i < len(pdf_files) - 1:  # Don't delay after the last file
                time.sleep(1)  # 1 second delay
        
        # Clear the progress bar and status text
        progress_bar.empty()
        status_text.empty()
        
        # Calculate overall compression ratio
        if results['total_size_original'] > 0:
            results['overall_compression_ratio'] = (
                (results['total_size_original'] - results['total_size_compressed']) / 
                results['total_size_original'] * 100
            )
        else:
            results['overall_compression_ratio'] = 0
        
        results['success'] = True
        results['message'] = f"Successfully processed {results['processed_files']} out of {results['total_files']} files."
        
        # Trigger reindexing if files were added
        if results['processed_files'] > 0:
            st.session_state.index_hash = ""
        
        return results
    
    except Exception as e:
        results['success'] = False
        results['message'] = f"Error importing PDFs from Google Drive: {str(e)}"
        import traceback
        results['error_details'] = traceback.format_exc()
        return results
    finally:
        # Ensure MongoDB client is closed
        if mongo_client:
            mongo_client.close()

# Main Streamlit application
def main():
    st.set_page_config(
        page_title="RAG- Kinetic Modeling",  # This changes the browser tab title
        page_icon="📚",  # This can be an emoji or a path to an image file
    )
    # Initialize session state
    if not initialize_session_state():
        st.stop()
    
    # Routing based on session state
    if st.session_state.current_page == 'login':
        login_page()
    elif st.session_state.current_page == 'admin_login':
        admin_login_form()
    elif st.session_state.logged_in:
        # Title
        st.title("Kinetic Modeling - RAG")
        
        # Logout button
        if st.sidebar.button("Logout"):
            # Reset session state
            for key in SESSION_KEYS:
                if hasattr(st.session_state, key):
                    delattr(st.session_state, key)
            st.session_state.current_page = 'login'
            st.rerun()
        
        # Welcome message
        st.write(f"Welcome {'Admin' if st.session_state.is_admin else 'User'}")
        
        # Check if we need to rerun
        if st.session_state.should_rerun:
            st.session_state.should_rerun = False
            st.rerun()
        
        # Knowledge base management (Admin only)
        if st.session_state.is_admin:
            st.sidebar.title("Knowledge Base Management")
            
            # Create tabs for PDF, URL, Google Drive and Embeddings
            tab1, tab2, tab3, tab4 = st.sidebar.tabs(["PDF Documents", "URLs", "Google Drive", "Embeddings"])
            
            with tab1:
                # Upload new PDF
                st.write("Upload a new PDF")
                uploaded_file = st.file_uploader("PDF Upload", type="pdf", key="file_uploader", 
                                                accept_multiple_files=False, label_visibility="collapsed")
                
                # Display success/error messages if they exist
                if st.session_state.upload_success_message:
                    st.success(st.session_state.upload_success_message)
                    # Clear the message after displaying once
                    st.session_state.upload_success_message = None
                    
                if st.session_state.delete_success_message:
                    st.success(st.session_state.delete_success_message)
                    st.session_state.delete_success_message = None
                    
                if st.session_state.delete_error_message:
                    st.error(st.session_state.delete_error_message)
                    st.session_state.delete_error_message = None
                
                # Handle file upload with proper state management
                if uploaded_file is not None and uploaded_file.name not in st.session_state.uploaded_files:
                    handle_file_upload(uploaded_file)
                
                # Display available PDFs
                st.write("Available PDFs:")
                
                # Add the Delete All Files button
                add_delete_all_button_to_tab1()
                
                # Delete confirmation dialog for individual PDFs
                if st.session_state.confirm_delete:
                    st.warning(f"Are you sure you want to delete {st.session_state.confirm_delete}?")
                    col1, col2 = st.columns(2)
                    if col1.button("Yes, Delete", key="confirm_yes"):
                        confirm_delete()
                    if col2.button("Cancel", key="confirm_no"):
                        cancel_delete()
                
                # Display PDF list with delete buttons
                for index, pdf in enumerate(st.session_state.uploaded_files):
                    col1, col2 = st.columns([3, 1])
                    col1.write(pdf)
                    # Use a unique key by adding the index
                    if col2.button("🗑️", key=f"delete_{index}_{pdf}", help="Delete this PDF"):
                        set_delete_confirmation(pdf)
            
            with tab2:
                # Add new URL
                st.write("Add a new URL")
                
                # Check if we need to clear the URL input
                if st.session_state.should_clear_url:
                    url_input_value = ""
                    st.session_state.should_clear_url = False
                else:
                    url_input_value = st.session_state.url_input if "url_input" in st.session_state else ""
                
                st.text_input("URL Input", value=url_input_value, key="url_input", placeholder="https://example.com", 
                             label_visibility="collapsed")
                if st.button("Add URL", key="add_url_button"):
                    add_url()
                             
                # Display URL success/error messages if they exist
                if st.session_state.url_delete_success_message:
                    st.success(st.session_state.url_delete_success_message)
                    st.session_state.url_delete_success_message = None
                    
                if st.session_state.url_delete_error_message:
                    st.error(st.session_state.url_delete_error_message)
                    st.session_state.url_delete_error_message = None
                
                # Display available URLs
                st.write("Available URLs:")
                
                # Delete confirmation dialog for URLs
                if st.session_state.confirm_delete_url:
                    st.warning(f"Are you sure you want to remove this URL?")
                    st.write(st.session_state.confirm_delete_url)
                    col1, col2 = st.columns(2)
                    if col1.button("Yes, Remove", key="confirm_url_yes"):
                        confirm_delete_url()
                    if col2.button("Cancel", key="confirm_url_no"):
                        cancel_delete()
                
                # Display URL list with delete buttons
                for url in st.session_state.urls:
                    col1, col2 = st.columns([3, 1])
                    
                    # Show shortened URL for display
                    display_url = url if len(url) < 30 else url[:27] + "..."
                    
                    # Use Streamlit's native link component instead of HTML
                    # This approach uses Streamlit's built-in link function
                    col1.write(f"[{display_url}]({url})")
                    
                    # Use a trash bin emoji for the delete button
                    if col2.button("🗑️", key=f"delete_url_{url}", help="Remove this URL"):
                        set_delete_url_confirmation(url)

            # New Google Drive Tab
            with tab3:
                st.write("Import PDFs from Google Drive")
                
                # Add info about the connected folder
                st.info(f"Connected to Google Drive folder ID: {GOOGLE_DRIVE_FOLDER_ID}")
                
                # Add import button
                if st.button("Import PDFs from Google Drive", key="import_gdrive"):
                    with st.spinner("Connecting to Google Drive..."):
                        import_results = import_pdfs_from_google_drive()
                        
                        if import_results['success']:
                            st.success(import_results['message'])
                            
                            # Display statistics
                            st.write("### Import Statistics")
                            
                            # Use columns for the stats display
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Files", import_results['total_files'])
                            with col2:
                                st.metric("Successfully Processed", import_results['processed_files'])
                            with col3:
                                st.metric("Errors", import_results['error_files'])
                            
                            if import_results['processed_files'] > 0:
                                # Show compression metrics
                                st.write("### Compression Results")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    original_size_mb = import_results['total_size_original'] / (1024 * 1024)
                                    compressed_size_mb = import_results['total_size_compressed'] / (1024 * 1024)
                                    st.metric("Original Size", f"{original_size_mb:.2f} MB")
                                with col2:
                                    st.metric("Compressed Size", f"{compressed_size_mb:.2f} MB", 
                                             delta=f"-{import_results['overall_compression_ratio']:.1f}%")
                                
                                # Show file details in an expander
                                with st.expander("View Processed Files"):
                                    for file_info in import_results['files']:
                                        if file_info['status'] == 'success':
                                            st.write(f"✅ **{file_info['original_filename']}** → **{file_info['standardized_filename']}**")
                                            st.write(f"   Compression: {file_info['compression_ratio']:.1f}% reduction")
                                        elif file_info['status'] == 'skipped':
                                            st.write(f"⏭️ **{file_info['original_filename']}** (skipped - already exists)")
                                        else:
                                            st.write(f"❌ **{file_info['original_filename']}** (error: {file_info.get('message', 'Unknown error')})")
                                        st.write("---")
                                        
                                # Automatically update the files list
                                st.session_state.uploaded_files = [
                                    file_doc["filename"] for file_doc in st.session_state.files_collection.find({}, {"filename": 1, "_id": 0})
                                ]

                                # Use experimental_rerun to refresh the entire page
                                st.rerun()
                        else:
                            st.error(import_results['message'])
                            if 'error_details' in import_results:
                                with st.expander("Error Details"):
                                    st.code(import_results['error_details'])
                
                # Display information about naming conventions
                with st.expander("PDF Naming Convention"):
                    st.markdown("""
                    ## Standard Naming Format
                    All PDF files follow this format:
                    **[Year]_[FirstAuthorLastName]_[PaperTitle]_[Category].pdf**
                    
                    ### Examples:
                    * 2023_Smith_Kinetic-Modeling-In-Alzheimers-Disease_Brain.pdf
                    * 2019_Johnson_Tracer-Kinetics-In-Lung-Tissue_Lung.pdf
                    * 2021_Wang_Advanced-Optimization-Algorithms-For-Kinetic-Analysis_Advanced.pdf
                    
                    ### Guidelines for Each Element
                    1. **Year [YYYY]**
                       * 4-digit year format (e.g., 2023)
                       * Unknown publication year uses the submission/preprint year followed by "pre" (e.g., 2023pre)
                       
                    2. **First Author's Last Name**
                       * Only last name of the first author
                       * Hyphenated last names include the hyphen (e.g., Smith-Jones)
                       * No first names or initials
                       
                    3. **Paper Title**
                       * For shorter titles, includes the full title with spaces replaced by hyphens
                       * For longer titles, includes only the first 5-6 significant words
                       * Articles (a, an, the) are removed when possible
                       
                    4. **Category**
                       * Main category (Brain, Lung, Liver, etc.)
                       * For multiple categories, uses the primary category
                       * Subcategories use format: Brain-Neuroinflammation
                    """)

            with tab4:
                st.write("Export Embeddings for Manifold Comparison")
                
                # Display information about the current index
                # Check if there are documents or URLs instead of checking the index directly
                if len(st.session_state.uploaded_files) > 0 or len(st.session_state.urls) > 0:
                    # If we're currently indexing, show that status
                    if st.session_state.indexing_status == "in_progress":
                        st.info("Indexing in progress. Please wait for indexing to complete before exporting.")
                    else:
                        # Get approximate size of the index if possible
                        try:
                            if st.session_state.index is not None and hasattr(st.session_state.index, "_docstore") and hasattr(st.session_state.index._docstore, "_nodes"):
                                node_count = len(st.session_state.index._docstore._nodes)
                                st.write(f"Current index contains approximately {node_count} nodes.")
                            else:
                                st.write(f"Index is available for export.")
                        except Exception:
                            st.write("Index is available for export.")
                        
                        # Add export button
                        if st.button("Export Embeddings", key="export_embeddings"):
                            with st.spinner("Exporting embeddings..."):
                                success, result = export_embeddings(st.session_state.index)
                                
                                # Inside the success block of the export function:
                                if success:
                                    st.success("Embeddings exported successfully! Choose a format to download:")
                                    
                                    # Use a container with custom CSS for better spacing
                                    st.markdown("""
                                    <style>
                                    .download-container {
                                        display: flex;
                                        gap: 30px;
                                        margin-top: 20px;
                                        margin-bottom: 20px;
                                    }
                                    .download-option {
                                        flex: 1;
                                        padding: 15px;
                                        border-radius: 5px;
                                        background-color: rgba(255, 255, 255, 0.05);
                                    }
                                    .download-title {
                                        font-size: 1.2rem;
                                        font-weight: bold;
                                        margin-bottom: 15px;
                                    }
                                    .format-benefits {
                                        margin-bottom: 15px;
                                    }
                                    </style>
                                    <div class="download-container">
                                        <div class="download-option">
                                            <div class="download-title">JSON Format</div>
                                            <div class="format-benefits">
                                                ✓ Human-readable<br>
                                                ✓ Works with any programming language<br>
                                                ✓ Easy to inspect and transform
                                            </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # JSON download link
                                    json_content, json_filename = result["json"]
                                    json_link = get_text_download_link(
                                        json_content, 
                                        json_filename, 
                                        "📥 Download JSON"
                                    )
                                    st.markdown(json_link, unsafe_allow_html=True)
                                    
                                    # Continue the HTML layout
                                    st.markdown("""
                                        </div>
                                        <div class="download-option">
                                            <div class="download-title">Pickle Format</div>
                                            <div class="format-benefits">
                                                ✓ Preserves numpy arrays exactly<br>
                                                ✓ Smaller file size for large datasets<br>
                                                ✓ Faster for Python-based analysis
                                            </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Pickle download link
                                    pickle_content, pickle_filename = result["pickle"]
                                    pickle_link = get_binary_download_link(
                                        pickle_content, 
                                        pickle_filename, 
                                        "📥 Download Pickle"
                                    )
                                    st.markdown(pickle_link, unsafe_allow_html=True)
                                    
                                    # Close the HTML container
                                    st.markdown("</div></div>", unsafe_allow_html=True)
                                    
                                    # Display additional info about the export
                                    with st.expander("Export Contents"):
                                        st.markdown("""
                                        The exported file contains:
                                        
                                        - **Metadata**: Timestamp, document counts, and context information
                                        - **Node Data**: For each chunk of text in your knowledge base
                                        - Unique node ID
                                        - Text content
                                        - Source metadata (document name, page, etc.)
                                        - Full embedding vector
                                        
                                        This structured format is ideal for:
                                        - Comparing embeddings between different knowledge bases
                                        - Visualizing embedding spaces using t-SNE or UMAP
                                        - Analyzing topic clustering across different domains
                                        - Tracking knowledge base evolution over time
                                        """)
                else:
                    st.warning("No index available. Please add documents and index them first.")            
            
            # Force reindex button (outside tabs)
            if st.sidebar.button("⟳ Reindex All", key="force_reindex", help="Force reindex all documents and URLs"):
                st.session_state.index_hash = ""  # Force reindex
                st.session_state.should_rerun = True
        
        # Current sources hash
        current_hash = get_sources_hash()
        
        # Check if reindexing is needed
        need_reindex = False
        
        # Track if sources have changed
        if current_hash != st.session_state.index_hash:
            need_reindex = True
            st.session_state.index_hash = current_hash
        
        # Show indexing status with st.spinner
        indexing_placeholder = st.empty()
        if st.session_state.indexing_status == "in_progress":
            with indexing_placeholder.container():
                with st.spinner("Reindexing knowledge base..."):
                    pass
        
        # Display source counts - only for admin
        pdf_count = len(st.session_state.uploaded_files)
        url_count = len(st.session_state.urls)
        if st.session_state.is_admin:
            st.write(f"Knowledge base: {pdf_count} PDFs and {url_count} URLs")
        
        # Load and index documents if needed
        if need_reindex or st.session_state.index is None:
            with indexing_placeholder.container():
                with st.spinner("Reindexing knowledge base..."):
                    st.session_state.index = load_and_index_documents()
                    st.session_state.last_update_time = time.time()
                        
        # Create query engine if index exists
        if st.session_state.index is not None:
            query_engine = create_optimized_query_engine(st.session_state.index)
        else:
            query_engine = None
            if pdf_count > 0 or url_count > 0:
                st.warning("Failed to index documents. Please check your sources and try again.")
            else:
                st.info("No sources found. Please add PDFs or URLs to start.")
        
        # Chat interface
        display_chat_interface(query_engine)
        
        # Add a footer with system information
        st.markdown("---")
        st.markdown(f"Last index update: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.session_state.last_update_time))}")
        if st.session_state.is_admin:
            mongo_status = "Connected" if hasattr(st.session_state, "files_collection") else "Disconnected"
            st.markdown(f"MongoDB Status: {mongo_status} | Index Status: {st.session_state.indexing_status}")    

# Run the application
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error(traceback.format_exc())    