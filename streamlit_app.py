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
import tempfile

import base64
import numpy as np
import json
import pickle
from datetime import datetime
from io import BytesIO

# Add these imports to your existing imports if not already present
import io
import re
import fitz  # PyMuPDF
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
import pikepdf
import openai
import uuid
from datetime import datetime, timedelta

# Add these constants to your existing constants
GOOGLE_DRIVE_FOLDER_ID = "1w-6V_XZvNK6gOeFT61KZk1GLHg65brKR"  # Your Google Drive folder ID
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

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
    "confirm_delete_all",
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
    "q1_clicked",
    "q2_clicked", 
    "q3_clicked",
    "question_text",
    "message_sent",
    "index_version_in_db",  # Track the version of the index stored in DB
    "index_loaded_from_db",  # Flag to track if we successfully loaded from DB 
    "is_collaborator",  # Flag to identify collaborator user type
    "collaborator_upload_success",  # Message for successful upload
    "collaborator_upload_error",    # Error message for failed upload
    "upload_preview_id",            # ID of upload being previewed
    "upload_action_error",          # Error during upload approval/denial actions
    "upload_action_success",        # Success message for upload actions
]



# Function to hash password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# 1. Constants for temporary uploads
TEMP_DB_NAME = "temp_collaborator_uploads"

# Login page
# 2. COLLABORATOR UPLOAD PAGE 
def collaborator_upload_page():
    """Page for collaborators to upload PDFs for review."""
    st.title("Collaborator PDF Upload")
    st.write("Upload your PDF for review by the admin team")
    
    # Initialize session state variables if they don't exist
    if "collaborator_upload_success" not in st.session_state:
        st.session_state.collaborator_upload_success = None
    if "collaborator_upload_error" not in st.session_state:
        st.session_state.collaborator_upload_error = None
    
    # Display any previous messages
    if st.session_state.collaborator_upload_success:
        st.success(st.session_state.collaborator_upload_success)
        st.session_state.collaborator_upload_success = None
    
    if st.session_state.collaborator_upload_error:
        st.error(st.session_state.collaborator_upload_error)
        st.session_state.collaborator_upload_error = None
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Show file info before upload
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        
        st.write("File details:")
        for key, value in file_details.items():
            st.write(f"- **{key}:** {value}")
        
        # Upload button with a dedicated action
        if st.button("Submit for Review", key="submit_for_review_btn"):
            # Generate a unique ID for the upload
            upload_id = str(uuid.uuid4())
            
            try:
                # Connect to temp database
                temp_client = pymongo.MongoClient(os.getenv("MONGO_URI"))
                temp_db = temp_client[TEMP_DB_NAME]
                temp_fs = gridfs.GridFS(temp_db)
                
                # Create pending_uploads collection if it doesn't exist
                if "pending_uploads" not in temp_db.list_collection_names():
                    temp_db.create_collection("pending_uploads")
                
                # Prepare metadata for temporary storage
                temp_upload_metadata = {
                    "_id": upload_id,
                    "filename": uploaded_file.name,
                    "upload_timestamp": datetime.now(),
                    "status": "pending",
                    "original_size": uploaded_file.size
                }
                
                # Store file in GridFS
                file_id = temp_fs.put(
                    uploaded_file.getvalue(), 
                    filename=uploaded_file.name,
                    metadata=temp_upload_metadata
                )
                
                # Store metadata separately
                temp_db.pending_uploads.insert_one({
                    **temp_upload_metadata,
                    "gridfs_id": file_id
                })
                
                # Set success message
                st.session_state.collaborator_upload_success = "PDF uploaded successfully and sent for review!"
                st.rerun()
                
            except Exception as e:
                st.session_state.collaborator_upload_error = f"Error uploading file: {str(e)}"
                st.rerun()
    
    # Add some info text
    st.markdown("""
    ### Upload Guidelines
    - Only PDF files are accepted
    - PDFs will be reviewed by administrators before being added to the main database
    - You will not receive notification when your PDF is approved - it will be available in the main system
    - Please ensure uploaded PDFs are relevant to kinetic modeling research
    """)
    
    # Add logout button
    if st.button("Return to Login", key="collab_logout_btn"):
        for key in SESSION_KEYS:
            if key in st.session_state:
                delattr(st.session_state, key)
        st.session_state.current_page = 'login'
        st.rerun()

def show_pdf_preview_improved(upload, file_content):
    """Improved PDF preview with wider layout."""
    # Create a container for the preview
    with st.expander("PDF Preview", expanded=True):
        # Open the PDF document first
        try:
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            
            # Add download button with a unique key
            download_key = f"download_{upload['_id']}"
            st.download_button(
                label="⬇️ Download PDF",
                data=file_content,
                file_name=upload['filename'],
                mime="application/pdf",
                key=download_key,
                use_container_width=True
            )

            # Add documentation-like display of file details
            st.write(f"**Filename:** {upload['filename']}")
            st.write(f"**Total pages:** {pdf_document.page_count}")
            st.write(f"**Size:** {upload['original_size'] / 1024:.2f} KB")
            
            # Display the first few pages as images
            # Only show first 3 pages to avoid overloading the interface
            max_preview_pages = min(3, pdf_document.page_count)
            
            # Use the full width of the container for better display
            for page_num in range(max_preview_pages):
                page = pdf_document[page_num]
                
                # Adjust the zoom factor for better visibility
                zoom_factor = 2.0  # Increase the zoom for better clarity
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom_factor, zoom_factor))
                img_bytes = pix.tobytes("png")
                
                # Add page indicator
                st.write(f"**Page {page_num + 1}/{pdf_document.page_count}**")
                
                # Display the image with full width
                st.image(
                    img_bytes, 
                    use_container_width=True
                )
                
                # Add a separator between pages
                if page_num < max_preview_pages - 1:
                    st.markdown("---")
            
            if pdf_document.page_count > max_preview_pages:
                st.info(f"Preview limited to first {max_preview_pages} pages. Download the PDF to view all {pdf_document.page_count} pages.")
            
        except Exception as e:
            st.error(f"Error rendering PDF preview: {str(e)}")
            st.write("PDF preview rendering failed. Please use the download button to view the file.")

# Also update the collaborator_uploads_review_tab function to handle rejection status
def collaborator_uploads_review_tab():
    """Tab in admin interface to review collaborator uploads."""
    st.write("## Collaborator PDF Uploads")
    
    try:
        # Connect to temp database
        mongo_client = pymongo.MongoClient(os.getenv("MONGO_URI"))
        temp_db = mongo_client[TEMP_DB_NAME]
        
        # Make sure the collection exists
        if "pending_uploads" not in temp_db.list_collection_names():
            temp_db.create_collection("pending_uploads")
        
        # Fetch pending uploads
        pending_uploads = list(temp_db.pending_uploads.find({"status": "pending"}))
        
        # Get recently approved/rejected uploads within the last 30 seconds
        thirty_seconds_ago = datetime.now() - timedelta(seconds=30)
        
        # Get approved uploads
        recently_approved_uploads = list(temp_db.pending_uploads.find({
            "status": "approved", 
            "approved_at": {"$gt": thirty_seconds_ago}
        }))
        
        # Get rejected uploads (due to duplicates)
        recently_rejected_uploads = list(temp_db.pending_uploads.find({
            "status": "rejected", 
            "rejected_at": {"$gt": thirty_seconds_ago},
            "rejection_reason": {"$in": ["duplicate_content", "duplicate_filename"]}
        }))
        
        # If no pending uploads and no recent approvals/rejections, show info
        if not pending_uploads and not recently_approved_uploads and not recently_rejected_uploads:
            st.info("No pending uploads to review")
            return
        
        # Show count of pending uploads
        if pending_uploads:
            st.write(f"Found {len(pending_uploads)} pending upload(s)")
        
        # Show recently approved uploads first with success message
        for upload in recently_approved_uploads:
            with st.container():
                # Show a success message for normal approval
                st.markdown(f"""
                <div style="padding: 10px; background-color: rgba(0, 200, 0, 0.1); border-left: 5px solid #00c000; margin-bottom: 15px;">
                    <h4 style="margin-top: 0; color: #00c000;">✅ Successfully Approved</h4>
                    <p><b>Original:</b> {upload['filename']}</p>
                    <p><b>Renamed to:</b> {upload.get('standardized_filename', 'Unknown')}</p>
                    <p><i>This file has been added to the database.</i></p>
                </div>
                """, unsafe_allow_html=True)
        
        # Show recently rejected uploads with error message
        for upload in recently_rejected_uploads:
            with st.container():
                duplicate_of = upload.get('duplicate_of', 'an existing file')
                reason = "content is identical to" if upload.get('rejection_reason') == 'duplicate_content' else "has the same standardized name as"
                
                # Show a rejection message
                st.markdown(f"""
                <div style="padding: 10px; background-color: rgba(255, 0, 0, 0.1); border-left: 5px solid #ff0000; margin-bottom: 15px;">
                    <h4 style="margin-top: 0; color: #ff0000;">❌ Upload Rejected - Duplicate File</h4>
                    <p><b>Original:</b> {upload['filename']}</p>
                    <p><b>Reason:</b> This file {reason} '{duplicate_of}'</p>
                    <p><i>The file was not added to the database to prevent duplication.</i></p>
                </div>
                """, unsafe_allow_html=True)
        
        # Set up state variables for confirmation dialogs if they don't exist
        if "confirm_approve_id" not in st.session_state:
            st.session_state.confirm_approve_id = None
        if "confirm_deny_id" not in st.session_state:
            st.session_state.confirm_deny_id = None
        
        # Process each pending upload
        for index, upload in enumerate(pending_uploads):
            with st.container():
                st.markdown("---")
                st.subheader(upload['filename'])
                
                # Display file details
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Upload Date:** {upload['upload_timestamp'].strftime('%Y-%m-%d %H:%M')}")
                with col2:
                    st.write(f"**Size:** {upload['original_size'] / 1024:.2f} KB")
                
                # Put the buttons in a centered layout with equal widths and consistent styling
                button_cols = st.columns(3)
                
                # Preload file content for preview
                temp_fs = gridfs.GridFS(temp_db)
                file_content = temp_fs.get(upload['gridfs_id']).read()
                
                # Unique key for each preview button using index
                unique_preview_key = f"preview_{index}_{upload['_id']}"
                unique_approve_key = f"approve_{index}_{upload['_id']}"
                unique_deny_key = f"deny_{index}_{upload['_id']}"
                
                # Preview Button
                with button_cols[0]:
                    if st.button("👁️ Preview", key=unique_preview_key, use_container_width=True):
                        # Dynamically create session state for file content with unique key
                        st.session_state[f"file_content_{unique_preview_key}"] = file_content
                        
                        preview_state_key = f"show_preview_{unique_preview_key}"
                        if preview_state_key not in st.session_state:
                            st.session_state[preview_state_key] = True
                        else:
                            st.session_state[preview_state_key] = not st.session_state[preview_state_key]
                        st.rerun()
                
                # Approve Button
                with button_cols[1]:
                    if st.button("✅ Approve", key=unique_approve_key, use_container_width=True):
                        st.session_state.confirm_approve_id = upload['_id']
                        st.session_state.confirm_deny_id = None
                        st.rerun()
                
                # Deny Button
                with button_cols[2]:
                    if st.button("❌ Deny", key=unique_deny_key, use_container_width=True):
                        st.session_state.confirm_deny_id = upload['_id']
                        st.session_state.confirm_approve_id = None
                        st.rerun()
                
                # Show preview if requested - with improved layout
                preview_state_key = f"show_preview_{unique_preview_key}"
                if preview_state_key in st.session_state and st.session_state[preview_state_key]:
                    show_pdf_preview_improved(
                        upload, 
                        st.session_state[f"file_content_{unique_preview_key}"]
                    )
                
                # Show confirmation dialog for approval
                if st.session_state.confirm_approve_id == upload['_id']:
                    st.warning(f"⚠️ Are you sure you want to approve '{upload['filename']}'?")
                    confirm_cols = st.columns([1, 1])
                    
                    with confirm_cols[0]:
                        if st.button("✓ Yes, Approve", key=f"confirm_approve_{upload['_id']}", use_container_width=True):
                            with st.spinner(f"Processing '{upload['filename']}'..."):
                                # Process the approval
                                try:
                                    standardized_filename = process_approval_with_feedback(upload, temp_db)
                                    
                                    if standardized_filename:
                                        # Clear the confirmation state
                                        st.session_state.confirm_approve_id = None
                                        st.rerun()
                                    else:
                                        # If returned None, it means the file was rejected as a duplicate
                                        # Clear the confirmation state and rerun to show the rejection message
                                        st.session_state.confirm_approve_id = None
                                        st.rerun()
                                except Exception as e:
                                    # Display MongoDB connection error
                                    st.error(f"Error uploading to database: {str(e)}")
                                    # Show specific error for SSL/connection issues
                                    if "SSL" in str(e) or "connect" in str(e).lower():
                                        st.error("Database connection error. Please try again later.")
                    
                    with confirm_cols[1]:
                        if st.button("✗ Cancel", key=f"cancel_approve_{upload['_id']}", use_container_width=True):
                            st.session_state.confirm_approve_id = None
                            st.rerun()
                
                # Show confirmation dialog for denial
                if st.session_state.confirm_deny_id == upload['_id']:
                    st.warning(f"⚠️ Are you sure you want to deny '{upload['filename']}'?")
                    confirm_cols = st.columns([1, 1])
                    
                    with confirm_cols[0]:
                        if st.button("✓ Yes, Deny", key=f"confirm_deny_{upload['_id']}", use_container_width=True):
                            with st.spinner(f"Removing '{upload['filename']}'..."):
                                success = process_denial(upload, temp_db)
                                if success:
                                    # Clear the confirmation state
                                    st.session_state.confirm_deny_id = None
                                    # Add short delay to show message
                                    time.sleep(1)
                                    st.rerun()
                    
                    with confirm_cols[1]:
                        if st.button("✗ Cancel", key=f"cancel_deny_{upload['_id']}", use_container_width=True):
                            st.session_state.confirm_deny_id = None
                            st.rerun()
    
    except Exception as e:
        st.error(f"Error loading collaborator uploads: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

# Process denial function (same as before)
def process_denial(upload, temp_db):
    """Process the denial of a collaborator upload."""
    try:
        # Get GridFS instance
        temp_fs = gridfs.GridFS(temp_db)
        
        # Delete file from GridFS
        temp_fs.delete(upload['gridfs_id'])
        
        # Delete metadata from collection
        temp_db.pending_uploads.delete_one({"_id": upload['_id']})
        
        return True
        
    except Exception as e:
        st.error(f"Error denying upload: {str(e)}")
        return False

# 7. UPDATE LOGIN PAGE TO INCLUDE COLLABORATOR OPTION
def login_page():
    """Updated login page with collaborator option."""
    st.markdown("<h1 style='text-align: center;'>Kinetic Modeling - RAG</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Access Portal</h2>", unsafe_allow_html=True)
    
    # Create a three-column layout
    col1, col2, col3 = st.columns(3)
    
    # Admin Login
    with col1:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("Admin Login", key="admin_login_btn", use_container_width=True):
            st.session_state.current_page = 'admin_login'
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    # User Access
    with col2:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("User Access", key="user_login_btn", use_container_width=True):
            st.session_state.logged_in = True
            st.session_state.is_admin = False
            st.session_state.is_collaborator = False
            st.session_state.current_page = 'query'
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Collaborator
    with col3:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("Collaborator", key="collaborator_login_btn", use_container_width=True):
            st.session_state.logged_in = True
            st.session_state.is_admin = False
            st.session_state.is_collaborator = True
            st.session_state.current_page = 'collaborator_upload'
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
        st.session_state.is_collaborator = False
        st.session_state.collaborator_upload_success = None
        st.session_state.collaborator_upload_error = None
        st.session_state.upload_preview_id = None
        st.session_state.upload_action_error = None
        st.session_state.upload_action_success = None
        # Add flag for MongoDB error handling
        st.session_state.skip_mongodb_save = False
        
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
            # Set the collections after db is defined
            st.session_state.urls_collection = db["urls"]
            st.session_state.files_collection = db["files"]
            st.session_state.index_collection = db["index"]  # Now db is defined
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

# Updated handle_file_upload function with local storage functionality removed
def handle_file_upload(uploaded_file):
    if uploaded_file is not None:
        try:
            # Check for duplicate before processing
            # 1. First check by filename
            if uploaded_file.name in st.session_state.uploaded_files:
                st.warning(f"File with name '{uploaded_file.name}' already exists. Skipping upload.")
                return False
                
            # 2. Check by content hash for duplicate detection even if filename is different
            file_content = uploaded_file.getbuffer()
            content_hash = hashlib.md5(file_content).hexdigest()
            
            # Check if this hash exists in any file metadata
            if hasattr(st.session_state, "files_collection"):
                existing_file = st.session_state.files_collection.find_one({"content_hash": content_hash})
                if existing_file:
                    st.warning(f"This file appears to be a duplicate of '{existing_file['filename']}'. Skipping upload.")
                    return False
            
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
                    
                    # Save to GridFS directly using bytes
                    file_id = fs.put(
                        file_content,
                        filename=uploaded_file.name,
                        content_type=uploaded_file.type,
                        chunkSize=1048576  # Use 1MB chunks to reduce timeouts
                    )
                    
                    # Save file metadata with content hash for deduplication
                    files_collection.insert_one({
                        "filename": uploaded_file.name,
                        "gridfs_id": file_id,
                        "size": len(file_content),
                        "content_hash": content_hash,  # Add hash for deduplication
                        "last_modified": time.time()
                    })
                    
                    success = True
                    
                    # Update the session state list
                    if uploaded_file.name not in st.session_state.uploaded_files:
                        st.session_state.uploaded_files.append(uploaded_file.name)
                    
                    st.session_state.upload_success_message = f"Uploaded: {uploaded_file.name}"
                    
                    # Force a reset of the index to include the new file
                    st.session_state.index_hash = ""
                    
                    # Close the MongoDB connection
                    mongo_client.close()
                    
                except Exception as e:
                    retry_count += 1
                    
                    if retry_count >= max_retries:
                        st.error(f"Error uploading to MongoDB after {max_retries} attempts. Upload failed.")
                        success = False
                    else:
                        # Wait before retrying
                        time.sleep(1)
                finally:
                    # Close the temporary connection
                    if 'mongo_client' in locals():
                        mongo_client.close()
            
            # Set flag for rerun instead of direct call
            if success:
                st.session_state.should_rerun = True
                return True
            else:
                return False
            
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    return False

# Also modify the load_and_index_documents function to remove local file dependency
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
            
        st.session_state.indexing_status = "in_progress"
        
        documents = []
        
        # Directly load from MongoDB
        for filename in st.session_state.uploaded_files:
            try:
                file_doc = st.session_state.files_collection.find_one({"filename": filename})
                if file_doc and "gridfs_id" in file_doc:
                    grid_out = st.session_state.fs.get(file_doc["gridfs_id"])
                    
                    # Create a temporary file for processing
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                        temp_file.write(grid_out.read())
                        temp_file_path = temp_file.name
                    
                    try:
                        # Process this file
                        reader = SimpleDirectoryReader(input_files=[temp_file_path])
                        pdf_documents = reader.load_data()
                        documents.extend(pdf_documents)
                    finally:
                        # Clean up the temporary file
                        if os.path.exists(temp_file_path):
                            os.unlink(temp_file_path)
            except Exception as e:
                st.warning(f"Error processing {filename}: {str(e)}")
        
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
    try:
        # Add some randomness instruction to the prompt
        system_prompt = """
        Based on the current content in the knowledge base, generate 5 diverse and interesting starter questions 
        that users might want to ask. The questions should:
        
        1. Be directly answerable from the CURRENT knowledge base content
        2. Cover different topics or aspects of the available documents
        3. Be concise (10 words or less) but specific enough to be meaningful
        4. Focus on different documents if possible, not just the most recent one
        5. Be in the form of questions (end with ?)

        IMPORTANT GUIDELINES:
        - DO NOT mention specific author names or paper titles in the questions
        - DO NOT reference specific page numbers or sections like "page 1" or "section 3.2"
        - Focus on the concepts and topics rather than the sources
        - Make questions accessible to someone who hasn't read the documents yet
        - Questions should be general enough that a new user would understand them
        
        Try to be creative and generate questions that explore different aspects of the knowledge base.
        
        Format the response as a numbered list with just the questions, no additional text.
        """

        # Get more questions than we need
        response = query_engine.query(system_prompt)
        
        # Parse the response to extract the questions
        suggested_questions = []
        for line in response.response.strip().split('\n'):
            if line.strip() and (line.strip()[0].isdigit() and line.strip()[1:3] in ['. ', '? ', ') ']):
                question = line.strip()[3:].strip()
                if question:
                    suggested_questions.append(question)
        
        # If we got enough questions, randomly select 3
        if len(suggested_questions) >= 4:
            import random
            return random.sample(suggested_questions, 3)
        
        # Otherwise use the fallbacks
        if len(suggested_questions) < 3:
            suggested_questions = [
                "What documents are currently in the knowledge base?",
                "What topics are covered in the available sources?",
                "Can you summarize the main content of current documents?"
            ]
        
        return suggested_questions
    
    except Exception:
        # Default fallbacks
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
        
        # Create session state variables to track button clicks if they don't exist
        if "q1_clicked" not in st.session_state:
            st.session_state.q1_clicked = False
        if "q2_clicked" not in st.session_state:
            st.session_state.q2_clicked = False
        if "q3_clicked" not in st.session_state:
            st.session_state.q3_clicked = False
            
        # Create a session state variable to hold the question text
        if "question_text" not in st.session_state:
            st.session_state.question_text = ""
        
        # Function to handle button clicks
        def handle_q1_click():
            st.session_state.q1_clicked = True
            st.session_state.question_text = suggested_questions[0]
        
        def handle_q2_click():
            st.session_state.q2_clicked = True
            st.session_state.question_text = suggested_questions[1]
        
        def handle_q3_click():
            st.session_state.q3_clicked = True
            st.session_state.question_text = suggested_questions[2]
        
        # Display clickable buttons for each suggested question
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.button(suggested_questions[0], key="suggested_q1", on_click=handle_q1_click)
                
        with col2:
            st.button(suggested_questions[1], key="suggested_q2", on_click=handle_q2_click)
                
        with col3:
            st.button(suggested_questions[2], key="suggested_q3", on_click=handle_q3_click)
    
    # Process question clicks at the end of the rendering cycle
    if hasattr(st.session_state, 'q1_clicked') and st.session_state.q1_clicked:
        question = st.session_state.question_text
        st.session_state.q1_clicked = False
        st.session_state.question_text = ""
        process_new_message(question, query_engine)
        st.rerun()
    elif hasattr(st.session_state, 'q2_clicked') and st.session_state.q2_clicked:
        question = st.session_state.question_text
        st.session_state.q2_clicked = False
        st.session_state.question_text = ""
        process_new_message(question, query_engine)
        st.rerun()
    elif hasattr(st.session_state, 'q3_clicked') and st.session_state.q3_clicked:
        question = st.session_state.question_text
        st.session_state.q3_clicked = False
        st.session_state.question_text = ""
        process_new_message(question, query_engine)
        st.rerun()
    
    # Add a callback for when the send button is pressed
    def handle_send():
        if "user_message" in st.session_state and st.session_state.user_message:
            # This is needed to make sure Streamlit reinitializes the text area with an empty value
            message = st.session_state.user_message
            # Use a flag to indicate message was sent instead of modifying the input directly
            if "message_sent" not in st.session_state:
                st.session_state.message_sent = False
            st.session_state.message_sent = True
            process_new_message(message, query_engine)
    
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
            
            # Reset text area if a message was just sent
            if hasattr(st.session_state, "message_sent") and st.session_state.message_sent:
                st.session_state.user_message = ""
                st.session_state.message_sent = False
                
            user_message = st.text_area("", 
                                      key="user_message", 
                                      placeholder=placeholder,
                                      label_visibility="collapsed")
        
        # Place the send button in the third column 
        with col3:
            send_pressed = st.button("Send", key="send_message", on_click=handle_send, use_container_width=True)
    
    # We don't need this anymore since we're using the on_click handler
    # if send_pressed and user_message:
    #     process_new_message(user_message, query_engine)
    #     st.rerun()

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

# Function to delete all PDFs
def delete_all_pdfs():
    try:
        # Remove all files from GridFS
        for file_doc in st.session_state.files_collection.find({}, {"gridfs_id": 1}):
            try:
                if "gridfs_id" in file_doc:
                    st.session_state.fs.delete(file_doc["gridfs_id"])
            except Exception as e:
                st.warning(f"Error deleting file from GridFS: {str(e)}")
        
        # Clear the files collection
        st.session_state.files_collection.delete_many({})
        
        # Remove files from temp directory
        for filename in os.listdir(st.session_state.data_dir):
            if filename.endswith(".pdf"):
                try:
                    os.remove(os.path.join(st.session_state.data_dir, filename))
                except Exception as e:
                    st.warning(f"Error removing file from temp directory: {str(e)}")
        
        # Clear the uploaded_files list
        st.session_state.uploaded_files = []
        
        # Force complete reindex
        st.session_state.index_hash = ""
        
        # Set success message
        st.session_state.delete_success_message = "All PDFs have been deleted"
        
        return True
    except Exception as e:
        st.session_state.delete_error_message = f"Error deleting all PDFs: {str(e)}"
        import traceback
        st.error(traceback.format_exc())
        return False

# Modify the save_index_to_mongodb function to handle connection errors better
def save_index_to_mongodb(index, hash_value):
    """Save the index to MongoDB for persistence with improved error handling"""
    try:
        # Check if we should skip DB save based on session state flag
        if hasattr(st.session_state, "skip_mongodb_save") and st.session_state.skip_mongodb_save:
            st.warning("Skipping MongoDB save due to previous errors. Index will be used locally only.")
            return True
            
        # Check if index is too large - estimate size before attempting to save
        try:
            # Serialize to a temp buffer to estimate size
            temp_buffer = BytesIO()
            pickle.dump(index, temp_buffer)
            estimated_size = len(temp_buffer.getvalue())
            
            # If over 15MB (MongoDB document limit is 16MB), don't attempt to save
            if estimated_size > 15 * 1024 * 1024:
                st.warning(f"Index is too large ({estimated_size / (1024*1024):.2f} MB) to save to MongoDB. Using locally only.")
                st.session_state.skip_mongodb_save = True
                return True
        except Exception as size_error:
            st.warning(f"Could not estimate index size: {str(size_error)}. Will attempt save anyway.")
        
        # Use increased timeouts for large uploads
        mongo_client = pymongo.MongoClient(
            os.getenv("MONGO_URI"),
            serverSelectionTimeoutMS=30000,  # 30 seconds
            connectTimeoutMS=30000,
            socketTimeoutMS=60000,  # 60 seconds for large uploads
            maxPoolSize=1,  # Limit connections during large uploads
            retryWrites=True
        )
        
        # Use a fresh GridFS instance with the new client
        db = mongo_client["rag_system"]
        fs = gridfs.GridFS(db)
        index_collection = db["index"]
        
        # Serialize the index to bytes with a progress indicator
        with st.spinner("Serializing index for database storage..."):
            index_bytes = BytesIO()
            pickle.dump(index, index_bytes)
            index_bytes.seek(0)
            data = index_bytes.getvalue()
        
        # Check if existing index exists
        existing_index = index_collection.find_one({})
        if existing_index and "gridfs_id" in existing_index:
            try:
                # Delete old index
                fs.delete(existing_index["gridfs_id"])
            except Exception as e:
                st.warning(f"Could not delete old index: {str(e)}. Continuing with new save...")
        
        # Use chunked upload for large files to prevent timeouts
        with st.spinner("Uploading index to MongoDB (this may take a while)..."):
            chunk_size = 1048576  # 1MB
            
            # Store new index with larger chunk size
            index_id = fs.put(
                data,
                filename="vector_index",
                content_type="application/octet-stream",
                chunkSize=chunk_size
            )
            
            # Update or insert metadata document
            index_collection.delete_many({})
            index_collection.insert_one({
                "gridfs_id": index_id,
                "hash": hash_value,
                "timestamp": time.time(),
                "doc_count": len(st.session_state.uploaded_files),
                "url_count": len(st.session_state.urls),
                "size": len(data)
            })
        
        # Update session state to track current version in DB
        st.session_state.index_version_in_db = hash_value
        st.session_state.skip_mongodb_save = False  # Reset flag if successful
        
        # Close the temporary MongoDB connection
        mongo_client.close()
        
        return True
        
    except Exception as e:
        st.error(f"Error saving index to MongoDB: {str(e)}")
        
        # Add specific handling for connection errors
        if "AutoReconnect" in str(type(e)) or "ConnectionAborted" in str(e) or "timeout" in str(e).lower():
            st.warning("Connection timeout detected. Setting flag to avoid future MongoDB saves in this session.")
            # Set a flag to avoid future attempts in this session
            st.session_state.skip_mongodb_save = True
            
            # Still return True to continue with local processing
            st.info("Index will be used locally but changes won't be saved to MongoDB in this session.")
            return True
        else:
            import traceback
            st.error(traceback.format_exc())
            return False

# Function to load index from MongoDB
def load_index_from_mongodb():
    """Load the index from MongoDB if available"""
    try:
        # Check if there's an index stored
        index_doc = st.session_state.index_collection.find_one({})
        if not index_doc or "gridfs_id" not in index_doc:
            return None, None
        
        # Retrieve from GridFS
        grid_out = st.session_state.fs.get(index_doc["gridfs_id"])
        index_bytes = grid_out.read()
        
        # Deserialize the index
        try:
            index = pickle.loads(index_bytes)
            st.session_state.index_loaded_from_db = True
            return index, index_doc["hash"]
        except Exception as e:
            st.warning(f"Error deserializing index: {str(e)}")
            return None, None
        
    except Exception as e:
        st.warning(f"Error loading index from MongoDB: {str(e)}")
        return None, None    

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

# Function to extract text from PDF for analysis
def extract_text_for_analysis(pdf_content, max_pages=10):
    """
    Extract text from PDF with improved robustness.
    
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
                    
                    # Extract text with standard method
                    page_text = page.get_text("text")
                    
                    # If standard extraction fails, try alternative method
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
            
            return extracted_text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

# Function to parse the original filename for fallback metadata
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
    
    # Try to extract author - look for common patterns
    author_patterns = [
        r'([A-Z][a-z]+),\s*[A-Z]',  # LastName, FirstInitial
        r'([A-Z][a-z]+)\s+et\s+al',  # LastName et al
        r'([A-Z][a-z]+)\s+and\s+',   # LastName and...
    ]
    
    for pattern in author_patterns:
        author_match = re.search(pattern, basename)
        if author_match:
            parsed["author"] = author_match.group(1)
            break
    
    return parsed

# Function to clean title for filename
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
    
    # Replace spaces with hyphens
    title = re.sub(r'[\s_]+', '-', title)
    
    # Take first 5-6 words if very long
    parts = title.split('-')
    if len(parts) > 6:
        title = '-'.join(parts[:6])
    
    return title

# Function to get standardized filename
def get_standardized_filename(original_filename, pdf_content):
    """Use OpenAI to generate a standardized filename based on PDF content."""
    try:
        # Extract text from the PDF for analysis
        text = extract_text_for_analysis(pdf_content)
        
        # Parse original filename for fallback information
        parsed_info = parse_original_filename(original_filename)
        
        # If no text extracted, use fallback
        if not text.strip():
            print(f"No text extracted from {original_filename}")
            return f"{parsed_info['year']}_{parsed_info['author']}_{parsed_info['title']}_{parsed_info['category']}.pdf"
        
        # Use OpenAI to analyze the content
        client = openai.OpenAI()
        
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": """
                    You are a metadata extraction specialist for academic papers. Analyze the provided PDF text and extract:
                    1. Publication year (4-digit format, e.g., 2023)
                    2. First author's last name only (e.g., Smith, no first names or initials)
                    3. Paper title (simplified, use hyphens instead of spaces)
                    4. Main category (Brain, Lung, Liver, Heart, Kidney, or Advanced)
                    
                    FORMAT YOUR RESPONSE AS JSON with these keys:
                    {
                      "year": "YYYY",
                      "author": "LastName",
                      "title": "Simplified-Title-With-Hyphens",
                      "category": "Category"
                    }
                    
                    For "Advanced" category, include kinetic modeling papers and algorithm/method papers.
                    
                    Important guidelines:
                    - ONLY respond with the JSON, no explanations
                    - If you're unsure about the year, extract it from the original filename or use the most recent year in the text
                    - Title should have 5-6 significant words connected by hyphens (no articles like "a", "an", "the")
                    - If paper is clearly about kinetic modeling, use "Advanced" category
                    """
                },
                {
                    "role": "user", 
                    "content": f"Original Filename: {original_filename}\n\nDocument Text:\n{text}"
                }
            ],
            temperature=0.3,
            max_tokens=200,
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        response_text = completion.choices[0].message.content.strip()
        metadata = json.loads(response_text)
        
        # Create the standardized filename
        year = metadata.get('year', parsed_info['year'])
        author = metadata.get('author', parsed_info['author'])
        title = metadata.get('title', parsed_info['title'])
        category = metadata.get('category', 'Advanced')
        
        # Create and clean the filename
        filename = f"{year}_{author}_{title}_{category}.pdf"
        return clean_filename(filename)
        
    except Exception as e:
        print(f"Error creating standardized filename: {e}")
        # Get fallback information from the original filename
        parsed_info = parse_original_filename(original_filename)
        return f"{parsed_info['year']}_{parsed_info['author']}_{parsed_info['title']}_{parsed_info['category']}.pdf"

# Function to clean up a filename
def clean_filename(filename):
    """Clean filename to ensure it meets requirements."""
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace spaces with hyphens
    filename = re.sub(r'\s+', '-', filename)
    # Ensure length is reasonable
    if len(filename) > 100:
        base, ext = os.path.splitext(filename)
        filename = base[:95] + ext
    return filename

# Function to compress PDF
def compress_pdf(pdf_content):
    """Compress PDF content to reduce file size."""
    try:
        # Create a BytesIO object for the input PDF
        input_buffer = io.BytesIO(pdf_content)
        
        # Create a BytesIO object for the output PDF
        output_buffer = io.BytesIO()
        
        # Open the input PDF with pikepdf
        with pikepdf.open(input_buffer) as pdf:
            # Save with compression settings
            # Only use supported parameters
            pdf.save(output_buffer, 
                    compress_streams=True,
                    object_stream_mode=pikepdf.ObjectStreamMode.generate,
                    linearize=True)
        
        # Get the compressed content
        compressed_content = output_buffer.getvalue()
        
        # Only return compressed version if it's smaller
        if len(compressed_content) < len(pdf_content):
            print(f"PDF Compression: {(len(pdf_content) - len(compressed_content)) / len(pdf_content) * 100:.2f}% reduction")
            return compressed_content
        
        # If no reduction, return original
        print("PDF compression did not reduce file size")
        return pdf_content
    
    except Exception as e:
        print(f"Error compressing PDF: {e}")
        # Return original content if compression fails
        return pdf_content

# Updated is_duplicate_content function that works with both raw and compressed content
def is_duplicate_content(file_content):
    """
    Check if file content already exists in the database by calculating and comparing hash.
    Returns (is_duplicate, existing_filename) tuple.
    """
    # Calculate MD5 hash of the UNCOMPRESSED content
    content_hash = hashlib.md5(file_content).hexdigest()
    
    # Also check if this might be a compressed version of an existing file
    compressed_content = compress_pdf(file_content)
    compressed_hash = hashlib.md5(compressed_content).hexdigest()
    
    # Check if either hash exists in database
    existing_file = st.session_state.files_collection.find_one({
        "$or": [
            {"original_content_hash": content_hash},  # Hash before compression
            {"content_hash": content_hash},           # Hash after compression from other method
            {"original_content_hash": compressed_hash},
            {"content_hash": compressed_hash}
        ]
    })
    
    if existing_file:
        return True, existing_file.get("filename", "an existing file")
    
    return False, None

# Update the process_approval_with_feedback function with consistent hash storage
def process_approval_with_feedback(upload, temp_db):
    """Process the approval of a collaborator upload and return the standardized filename."""
    try:
        # MongoDB connection with improved parameters
        mongo_client = pymongo.MongoClient(
            os.getenv("MONGO_URI"),
            serverSelectionTimeoutMS=30000, 
            connectTimeoutMS=30000,
            socketTimeoutMS=60000,  
            maxPoolSize=1,
            retryWrites=True
        )
        
        main_db = mongo_client["rag_system"]
        
        # Get GridFS instances
        temp_fs = gridfs.GridFS(temp_db)
        main_fs = gridfs.GridFS(main_db)
        
        # Get file content
        file_content = temp_fs.get(upload['gridfs_id']).read()
        
        # Calculate original content hash before any processing
        original_content_hash = hashlib.md5(file_content).hexdigest()
        
        # Check for duplicate content
        is_duplicate, duplicate_filename = is_duplicate_content(file_content)
        
        if is_duplicate:
            st.error(f"This file is a duplicate of '{duplicate_filename}' that already exists in the database.")
            
            # Mark as rejected in the pending_uploads collection
            temp_db.pending_uploads.update_one(
                {"_id": upload['_id']},
                {"$set": {
                    "status": "rejected",
                    "rejected_at": datetime.now(),
                    "rejection_reason": "duplicate_content",
                    "duplicate_of": duplicate_filename
                }}
            )
            
            # Close MongoDB connection
            mongo_client.close()
            
            # Return None to indicate that the process should be stopped
            return None
        
        # Process the file (standardize name, compress)
        standardized_filename = get_standardized_filename(
            upload['filename'], 
            file_content
        )
        
        # Check by filename to avoid duplicates
        existing_by_name = main_db.files.find_one({"filename": standardized_filename})
        if existing_by_name:
            st.error(f"A file with the same standardized name already exists in the database.")
            
            # Mark as rejected in the pending_uploads collection
            temp_db.pending_uploads.update_one(
                {"_id": upload['_id']},
                {"$set": {
                    "status": "rejected",
                    "rejected_at": datetime.now(),
                    "rejection_reason": "duplicate_filename",
                    "duplicate_of": existing_by_name['filename']
                }}
            )
            
            # Close MongoDB connection
            mongo_client.close()
            
            # Return None to indicate that the process should be stopped
            return None
        
        # Compress the PDF
        compressed_content = compress_pdf(file_content)
        
        # Calculate compression stats
        original_size = len(file_content)
        compressed_size = len(compressed_content)
        compression_ratio = ((original_size - compressed_size) / original_size) * 100 if original_size > 0 else 0
        
        # Calculate content hash of the final compressed file
        content_hash = hashlib.md5(compressed_content).hexdigest()
        
        # Save to main GridFS
        main_file_id = main_fs.put(
            compressed_content,
            filename=standardized_filename,
            content_type='application/pdf',
            original_filename=upload['filename']
        )
        
        # Save metadata to main files collection WITH BOTH HASH VALUES
        main_db.files.insert_one({
            "filename": standardized_filename,
            "original_filename": upload['filename'],
            "gridfs_id": main_file_id,
            "size": compressed_size,
            "original_size": original_size,
            "compression_ratio": compression_ratio,
            "original_content_hash": original_content_hash,  # Hash BEFORE compression
            "content_hash": content_hash,                    # Hash AFTER compression
            "source": "collaborator_upload",
            "upload_id": upload['_id'],
            "last_modified": time.time()
        })
        
        # Update the session state list of files
        if standardized_filename not in st.session_state.uploaded_files:
            st.session_state.uploaded_files.append(standardized_filename)
        
        # Force a reset of the index to include the new file
        st.session_state.index_hash = ""
        
        # Store the rename info for feedback with timestamp
        temp_db.pending_uploads.update_one(
            {"_id": upload['_id']},
            {"$set": {
                "status": "approved",
                "approved_at": datetime.now(),
                "standardized_filename": standardized_filename
            }}
        )
        
        # Set a flag to ensure files are refreshed in all tabs
        if "files_refreshed" not in st.session_state:
            st.session_state.files_refreshed = False
        st.session_state.files_refreshed = True
        
        # Close MongoDB connection
        mongo_client.close()
        
        return standardized_filename
        
    except Exception as e:
        st.error(f"Error approving upload: {str(e)}")
        
        import traceback
        st.error(traceback.format_exc())
        return None

# Update the download_and_process_pdf function with consistent hash storage
def download_and_process_pdf(service, file_id, original_filename):
    """Download a PDF file from Google Drive, standardize name, and compress."""
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
        
        # Calculate hash BEFORE compression
        original_content_hash = hashlib.md5(file_content).hexdigest()
        
        # Check for duplicate content
        is_duplicate, duplicate_filename = is_duplicate_content(file_content)
        
        if is_duplicate:
            st.warning(f"File '{original_filename}' is a duplicate of '{duplicate_filename}' that already exists in the database.")
            return None
        
        # Get standardized filename
        standardized_filename = get_standardized_filename(original_filename, file_content)
        
        # Compress the PDF
        compressed_content = compress_pdf(file_content)
        
        # Calculate hash AFTER compression
        content_hash = hashlib.md5(compressed_content).hexdigest()
        
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
            'compression_ratio': compression_ratio,
            'original_content_hash': original_content_hash,  # Hash BEFORE compression
            'content_hash': content_hash                     # Hash AFTER compression
        }
    except Exception as e:
        print(f"Error downloading/processing file {original_filename}: {e}")
        return None

# Update the import_pdfs_from_google_drive function with consistent hash storage
def import_pdfs_from_google_drive():
    """Import PDFs from Google Drive, process them, and save to MongoDB."""
    results = {
        'success': False,
        'total_files': 0,
        'processed_files': 0,
        'skipped_files': 0,
        'error_files': 0,
        'total_size_original': 0,
        'total_size_compressed': 0,
        'files': []
    }
    
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
        
        # Add a progress bar for all files
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Track new files found
        new_files = []
        
        # Process each PDF file
        for i, pdf_file in enumerate(pdf_files):
            file_id = pdf_file['id']
            original_filename = pdf_file['name']
            
            # Update progress
            progress_percent = (i + 1) / len(pdf_files)
            progress_bar.progress(progress_percent)
            status_text.text(f"Checking file {i+1} of {len(pdf_files)}: {original_filename}")
            
            # Check if already in database by Drive ID
            drive_id_match = st.session_state.files_collection.find_one({"drive_file_id": file_id})
            if drive_id_match:
                results['skipped_files'] += 1
                results['files'].append({
                    'original_filename': original_filename,
                    'status': 'skipped',
                    'message': f'File from same Google Drive source already exists as {drive_id_match["filename"]}'
                })
                continue
            
            # Download and process the PDF - duplicate check happens inside this function
            processed_file = download_and_process_pdf(service, file_id, original_filename)
            
            if processed_file:
                new_files.append((processed_file, file_id))  # Store the pair of processed file and Drive ID
            else:
                # Already marked as duplicate in download_and_process_pdf function
                results['skipped_files'] += 1
                results['files'].append({
                    'original_filename': original_filename,
                    'status': 'skipped',
                    'message': 'File is a duplicate of an existing file'
                })
        
        # Clear the progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Message if no new files
        if not new_files:
            st.info("No new PDF files to import. All files already exist in the database.")
            results['message'] = "All files already exist in the database."
            results['success'] = True
            return results
        
        # Show how many new files we'll process
        st.info(f"Found {len(new_files)} new PDF files to import out of {len(pdf_files)} total.")
        
        # Add a progress bar for saving new files
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Save the new files to MongoDB
        for i, (processed_file, drive_id) in enumerate(new_files):
            # Update progress
            progress_percent = (i + 1) / len(new_files)
            progress_bar.progress(progress_percent)
            status_text.text(f"Saving file {i+1} of {len(new_files)}: {processed_file['standardized_filename']}")
            
            try:
                # Save to GridFS
                gridfs_file_id = st.session_state.fs.put(
                    processed_file['content'],
                    filename=processed_file['standardized_filename'],
                    content_type='application/pdf',
                    original_filename=processed_file['original_filename']
                )
                
                # Save file metadata to MongoDB WITH BOTH HASH VALUES
                st.session_state.files_collection.insert_one({
                    "filename": processed_file['standardized_filename'],
                    "original_filename": processed_file['original_filename'],
                    "gridfs_id": gridfs_file_id,
                    "original_content_hash": processed_file['original_content_hash'],  # Hash BEFORE compression
                    "content_hash": processed_file['content_hash'],                    # Hash AFTER compression
                    "size": processed_file['compressed_size'],
                    "original_size": processed_file['original_size'],
                    "compression_ratio": processed_file['compression_ratio'],
                    "source": "google_drive",
                    "drive_file_id": drive_id,  # Add Drive ID for future duplicate checks
                    "last_modified": time.time()
                })
                
                # Add to session state uploaded files
                if processed_file['standardized_filename'] not in st.session_state.uploaded_files:
                    st.session_state.uploaded_files.append(processed_file['standardized_filename'])
                
                # Update results statistics
                results['processed_files'] += 1
                results['total_size_original'] += processed_file['original_size']
                results['total_size_compressed'] += processed_file['compressed_size']
                results['files'].append({
                    'original_filename': processed_file['original_filename'],
                    'standardized_filename': processed_file['standardized_filename'],
                    'status': 'success',
                    'compression_ratio': processed_file['compression_ratio']
                })
            except Exception as insert_error:
                # Record error during MongoDB insertion
                results['error_files'] += 1
                results['files'].append({
                    'original_filename': processed_file['original_filename'],
                    'status': 'error',
                    'message': f'Failed to insert file: {str(insert_error)}'
                })
        
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
        
        # Create appropriate message based on what happened
        if results['processed_files'] > 0:
            results['message'] = f"Successfully processed {results['processed_files']} new files. Skipped {results['skipped_files']} existing files."
        else:
            results['message'] = f"No new files were processed. Skipped {results['skipped_files']} existing files."
        
        # Trigger reindexing if files were added
        if results['processed_files'] > 0:
            st.session_state.index_hash = ""
            
            # Set flag to trigger a UI refresh
            st.session_state.files_refreshed = True
        
        return results
    
    except Exception as e:
        results['success'] = False

        results['message'] = f"Error importing PDFs from Google Drive: {str(e)}"
        import traceback
        results['error_details'] = traceback.format_exc()
        return results

# Update the Google Drive tab function to ensure proper refreshing of files in all tabs
def google_drive_tab():
    st.write("Import PDFs from Google Drive")
    
    # Add info about the connected folder
    st.info(f"Connected to Google Drive folder ID: {GOOGLE_DRIVE_FOLDER_ID}")
    
    # Add import button
    if st.button("Import PDFs from Google Drive", key="import_gdrive"):
        with st.spinner("Connecting to Google Drive..."):
            import_results = import_pdfs_from_google_drive()
            
            if import_results['success']:
                # Different display based on what happened
                if import_results['processed_files'] > 0:
                    st.success(f"Successfully processed {import_results['processed_files']} new files. Skipped {import_results['skipped_files']} existing files.")
                    
                    # Display statistics
                    st.write("### Import Statistics")
                    
                    # Use columns for the stats display
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Files", import_results['total_files'])
                    with col2:
                        st.metric("New Files Processed", import_results['processed_files'])
                    with col3:
                        st.metric("Existing Files Skipped", import_results['skipped_files'])
                    with col4:
                        st.metric("Errors", import_results['error_files'])
                    
                    # Show compression metrics
                    if import_results['processed_files'] > 0:
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
                        # First show successful files
                        st.write("#### Successfully Processed:")
                        success_count = 0
                        for file_info in import_results['files']:
                            if file_info['status'] == 'success':
                                success_count += 1
                                st.write(f"✅ **{file_info['original_filename']}** → **{file_info['standardized_filename']}**")
                                st.write(f"   Compression: {file_info['compression_ratio']:.1f}% reduction")
                                st.write("---")
                        
                        if success_count == 0:
                            st.write("No files were successfully processed.")
                        
                        # Then show skipped files
                        st.write("#### Skipped (Already Exist):")
                        skip_count = 0
                        for file_info in import_results['files']:
                            if file_info['status'] == 'skipped':
                                skip_count += 1
                                st.write(f"⏭️ **{file_info['original_filename']}** ({file_info.get('message', 'already exists')})")
                        
                        if skip_count == 0:
                            st.write("No files were skipped.")
                            
                        # Finally show error files
                        if import_results['error_files'] > 0:
                            st.write("#### Errors:")
                            for file_info in import_results['files']:
                                if file_info['status'] == 'error':
                                    st.write(f"❌ **{file_info['original_filename']}** (error: {file_info.get('message', 'Unknown error')})")
                    
                    # Critical fix: Force a complete refresh of uploaded files list
                    if import_results['processed_files'] > 0:
                        # First update from MongoDB to get the latest files
                        st.session_state.uploaded_files = [
                            file_doc["filename"] for file_doc in st.session_state.files_collection.find({}, {"filename": 1, "_id": 0})
                        ]
                        
                        # Force reindex of the knowledge base
                        st.session_state.index_hash = ""
                        
                        # Add a flag to indicate we should refresh the UI
                        st.session_state.files_refreshed = True
                        
                        # Create a button to refresh the UI - when clicked, it will force a full rerun
                        if st.button("Refresh Files List and Reindex", key="refresh_all"):
                            st.rerun()
                        else:
                            # Automatically trigger a rerun to refresh everything
                            st.rerun()
                else:
                    # No new files were processed
                    st.info(import_results['message'])
            else:
                st.error(import_results['message'])
                if 'error_details' in import_results:
                    with st.expander("Error Details"):
                        st.code(import_results['error_details'])

# Main Streamlit application
def main():

    st.set_page_config(
        page_title="RAG- Kinetic Modeling",  # This changes the browser tab title
        page_icon="📚",  # This can be an emoji or a path to an image file
    )
    # Initialize session state
    if not initialize_session_state():
        st.stop()

    # Add a files_refreshed flag if it doesn't exist
    if "files_refreshed" not in st.session_state:
        st.session_state.files_refreshed = False    
    
    # Routing based on session state
    if st.session_state.current_page == 'login':
        login_page()
    elif st.session_state.current_page == 'admin_login':
        admin_login_form()
    elif st.session_state.current_page == 'collaborator_upload':
        collaborator_upload_page()
    elif st.session_state.logged_in:
        # If we've just refreshed the files, make sure the list is up to date
        if st.session_state.files_refreshed:
            # Refresh uploaded files from MongoDB
            st.session_state.uploaded_files = [
                file_doc["filename"] for file_doc in st.session_state.files_collection.find({}, {"filename": 1, "_id": 0})
            ]
            # Reset the flag
            st.session_state.files_refreshed = False

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
            tab1, tab2, tab3, tab4, tab5 = st.sidebar.tabs(["PDF Documents", "URLs", "Embeddings", "Google Drive", "Collaborator Uploads"])
            
            with tab1:
                st.write("Upload a new PDF")
                uploaded_file = st.file_uploader("PDF Upload", type="pdf", key="file_uploader", 
                                                accept_multiple_files=False, label_visibility="collapsed")
                
                # Display success/error messages if they exist
                if st.session_state.upload_success_message:
                    st.success(st.session_state.upload_success_message)
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
                
                # Add another divider before the PDF list
                st.markdown("---")

                # Add "Delete All" button at the top if PDFs exist
                if st.session_state.uploaded_files:
                    if st.button("🗑️ Delete All PDFs", key="delete_all_pdfs", help="Delete all PDFs"):
                        st.session_state.confirm_delete_all = True
                        st.rerun()
                
                # Add confirmation dialog for "Delete All PDFs"
                if "confirm_delete_all" not in st.session_state:
                    st.session_state.confirm_delete_all = False
                    
                if st.session_state.confirm_delete_all:
                    st.warning("⚠️ Are you sure you want to delete ALL PDFs? This action cannot be undone.")
                    col1, col2 = st.columns(2)
                    if col1.button("Yes, Delete All", key="confirm_all_yes"):
                        if delete_all_pdfs():
                            st.session_state.confirm_delete_all = False
                            st.rerun()
                    if col2.button("Cancel", key="confirm_all_no"):
                        st.session_state.confirm_delete_all = False
                        st.rerun()
                
                # Add divider after delete all option
                if st.session_state.uploaded_files:
                    st.markdown("---")

                # Add search box for PDFs
                search_query = st.text_input("Search PDFs", placeholder="Enter filename to search...", key="pdf_search_input")
                
                # Display available PDFs heading
                st.write("Available PDFs:")    
                    
                # Delete confirmation dialog for individual PDFs
                if st.session_state.confirm_delete:
                    st.warning(f"Are you sure you want to delete {st.session_state.confirm_delete}?")
                    col1, col2 = st.columns(2)
                    if col1.button("Yes, Delete", key="confirm_yes"):
                        confirm_delete()
                    if col2.button("Cancel", key="confirm_no"):
                        cancel_delete()
                
                # Filter PDFs based on search query
                filtered_pdfs = [
                    pdf for pdf in st.session_state.uploaded_files 
                    if search_query.lower() in pdf.lower()
                ]
                
                # Display PDF list with delete buttons
                for i, pdf in enumerate(filtered_pdfs):
                    col1, col2 = st.columns([3, 1])
                    col1.write(pdf)
                    # Use a unique key by adding an index to prevent duplicates
                    safe_pdf = pdf.replace(".", "_").replace(" ", "_").replace("-", "_")
                    if col2.button("🗑️", key=f"delete_{i}_{safe_pdf[:20]}", help="Delete this PDF"):
                        set_delete_confirmation(pdf)
                
                # Show message if no PDFs match the search
                if not filtered_pdfs and search_query:
                    st.info(f"No PDFs found matching '{search_query}'")
                elif not filtered_pdfs and not search_query and not st.session_state.uploaded_files:
                    st.info("No PDFs uploaded yet")       
            
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
                for i, url in enumerate(st.session_state.urls):
                    col1, col2 = st.columns([3, 1])
                    
                    # Show shortened URL for display
                    display_url = url if len(url) < 30 else url[:27] + "..."
                    
                    # Use Streamlit's native link component instead of HTML
                    col1.write(f"[{display_url}]({url})")
                    
                    # Use a unique key with index to prevent duplicates
                    # Use safe characters only
                    safe_url = url.replace("://", "_").replace(".", "_").replace("/", "_")
                    if col2.button("🗑️", key=f"delete_url_{i}_{safe_url[:10]}", help="Remove this URL"):
                        set_delete_url_confirmation(url)

            with tab3:
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

            with tab4:
                google_drive_tab()    

            with tab5:
                collaborator_uploads_review_tab()                 
            
            # Force reindex button (outside tabs)
            if st.sidebar.button("⟳ Reindex All", key="force_reindex", help="Force reindex all documents and URLs"):
                st.session_state.index_hash = ""  # Force reindex
                st.session_state.should_rerun = True
        
        

        # Initialize tracking variables if they don't exist
        if "index_version_in_db" not in st.session_state:
            st.session_state.index_version_in_db = ""
        if "index_loaded_from_db" not in st.session_state:
            st.session_state.index_loaded_from_db = False

        # Current sources hash
        current_hash = get_sources_hash()

        # Check if reindexing is needed
        need_reindex = False

        # If index is None, try to load from MongoDB first
        if st.session_state.index is None:
            loaded_index, loaded_hash = load_index_from_mongodb()
            if loaded_index is not None and loaded_hash == current_hash:
                st.session_state.index = loaded_index
                st.session_state.index_hash = loaded_hash
                st.success("Index loaded from database successfully")
            else:
                need_reindex = True
        # Otherwise check if sources have changed
        elif current_hash != st.session_state.index_hash:
            need_reindex = True

        # Only update the index hash if we're going to reindex
        if need_reindex:
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
        if need_reindex:
            with indexing_placeholder.container():
                with st.spinner("Reindexing knowledge base..."):
                    st.session_state.index = load_and_index_documents()
                    st.session_state.last_update_time = time.time()
                    
                    # Save the new index to MongoDB
                    if st.session_state.index is not None:
                        if save_index_to_mongodb(st.session_state.index, current_hash):
                            st.success("Index saved to database")
                        
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