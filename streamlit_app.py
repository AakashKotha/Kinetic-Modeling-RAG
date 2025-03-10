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
import pymongo
import gridfs
import hashlib

# Load environment variables
load_dotenv()

# Predefined admin credentials
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
    "upload_success_message"
]

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
    
    username = st.text_input("Username", key="admin_username")
    password = st.text_input("Password", type="password", key="admin_password")
    
    if st.button("Login", key="admin_login_submit"):
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.session_state.logged_in = True
            st.session_state.is_admin = True
            st.session_state.current_page = 'query'
            st.rerun()
        else:
            st.error("Invalid username or password")
    
    # Back button
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
                serverSelectionTimeoutMS=15000,
                connectTimeoutMS=15000,
                socketTimeoutMS=15000,
                retryWrites=True,
                tlsAllowInvalidCertificates=True  # For Atlas compatibility
            )
            
            mongo_client.admin.command('ping')
            
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
            
        except Exception as e:
            st.error(f"Error connecting to MongoDB: {str(e)}")
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
                        serverSelectionTimeoutMS=10000,  # Increase timeout to 10 seconds
                        connectTimeoutMS=10000,
                        socketTimeoutMS=30000,  # Increase socket timeout for large files
                        maxPoolSize=1,  # Limit pool size for uploads
                        retryWrites=True  # Enable retry writes
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
        # Optimize chunk size and overlap
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
    retriever = VectorIndexRetriever(index=index, similarity_top_k=4)
    return RetrieverQueryEngine(retriever=retriever)

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

# Function to confirm deletion of file
def confirm_delete():
    if st.session_state.confirm_delete:
        filename = st.session_state.confirm_delete
        
        try:
            # Remove from GridFS
            file_doc = st.session_state.files_collection.find_one({"filename": filename})
            if file_doc and "gridfs_id" in file_doc:
                st.session_state.fs.delete(file_doc["gridfs_id"])
                
            # Remove from metadata collection
            st.session_state.files_collection.delete_one({"filename": filename})
            
            # Remove from temp directory if exists
            temp_file_path = os.path.join(st.session_state.data_dir, filename)
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
            st.session_state.index_hash = ""
            if filename in st.session_state.uploaded_files:
                st.session_state.uploaded_files.remove(filename)
            
            # Store a success message in session state
            st.session_state.delete_success_message = f"Deleted: {filename}"
        except Exception as e:
            st.session_state.delete_error_message = f"Error deleting file: {str(e)}"
            
        # Clear the confirmation
        st.session_state.confirm_delete = None
        # Force direct rerun since this is triggered by a button click
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
            
            # Create tabs for PDF and URL management
            tab1, tab2 = st.sidebar.tabs(["PDF Documents", "URLs"])
            
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
                
                # Delete confirmation dialog for PDFs
                if st.session_state.confirm_delete:
                    st.warning(f"Are you sure you want to delete {st.session_state.confirm_delete}?")
                    col1, col2 = st.columns(2)
                    if col1.button("Yes, Delete", key="confirm_yes"):
                        confirm_delete()
                    if col2.button("Cancel", key="confirm_no"):
                        cancel_delete()
                
                # Display PDF list with delete buttons
                for pdf in st.session_state.uploaded_files:
                    col1, col2 = st.columns([3, 1])
                    col1.write(pdf)
                    # Use a trash bin emoji for the delete button
                    if col2.button("🗑️", key=f"delete_{pdf}", help="Delete this PDF"):
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
                    # Show shortened URL if too long
                    display_url = url if len(url) < 30 else url[:27] + "..."
                    col1.write(display_url)
                    # Use a trash bin emoji for the delete button
                    if col2.button("🗑️", key=f"delete_url_{url}", help="Remove this URL"):
                        set_delete_url_confirmation(url)
            
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
        
        # Display source counts
        pdf_count = len(st.session_state.uploaded_files)
        url_count = len(st.session_state.urls)
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
        
        # Query input
        query = st.text_input("Enter your query:", key="query_input")
        
        if st.button("Submit Query", key="submit_button"):
            if not query:
                st.warning("Please enter a query.")
            elif query_engine is None:
                st.error("Query engine is not available. Please check if your knowledge base is empty.")
            else:
                with st.spinner("Processing query..."):
                    try:
                        response = query_engine.query(query)
                        
                        if response.response:
                            st.write("Response:")
                            st.write(response.response)
                        else:
                            st.warning("No relevant answer could be found for your query.")
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
        
        # Reset indexing status after UI rendering is complete
        if st.session_state.indexing_status == "complete":
            st.session_state.indexing_status = "idle"

# Run the application
if __name__ == "__main__":
    main()