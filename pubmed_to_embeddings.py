"""
Enhanced PubMed to Embeddings Module
Complete implementation with PubMedBERT embeddings and downloadable files
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
import time
import re
import threading
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import io
import os
from sentence_transformers import SentenceTransformer
import torch

# Global variable to store results for download
if 'scraping_results' not in st.session_state:
    st.session_state.scraping_results = None
if 'embedding_results' not in st.session_state:
    st.session_state.embedding_results = None

def build_pubmed_url(keyword, time_filter="all", start_year=None, end_year=None, page=1):
    """Build PubMed URL with proper filters"""
    # Base URL with keyword and abstract filter
    base_url = f"https://pubmed.ncbi.nlm.nih.gov/?term={keyword.replace(' ', '+')}&filter=simsearch1.fha"
    
    # Add date filters
    if time_filter == "1yr":
        base_url += "&filter=datesearch.y_1"
    elif time_filter == "5yr":
        base_url += "&filter=datesearch.y_5"
    elif time_filter == "10yr":
        base_url += "&filter=datesearch.y_10"
    elif time_filter == "custom" and start_year and end_year:
        base_url += f"&filter=dates.{start_year}-{end_year}"
    
    # Add page parameter
    if page > 1:
        base_url += f"&page={page}"
    
    return base_url

def get_total_pages(search_url):
    """Extract total number of pages from search results"""
    try:
        response = requests.get(search_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the "of X" label
        total_pages_label = soup.select_one('label.of-total-pages')
        if total_pages_label:
            # Extract number from "of 21,987" text (handle comma-separated numbers)
            text = total_pages_label.text.strip()
            match = re.search(r'of ([\d,]+)', text)
            if match:
                # Remove commas and convert to int
                pages_str = match.group(1).replace(',', '')
                return int(pages_str)
        
        return 1  # Default to 1 page if not found
    except Exception as e:
        print(f"Error getting total pages: {e}")
        return 1

def get_article_urls(search_url):
    """Extract article URLs from the search results page"""
    try:
        response = requests.get(search_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        article_links = []
        for link in soup.select('a.docsum-title'):
            article_url = "https://pubmed.ncbi.nlm.nih.gov" + link['href']
            article_links.append(article_url)
        
        return article_links
    except Exception as e:
        print(f"Error getting article URLs: {e}")
        return []

def get_total_articles_count(keyword, time_filter, start_year=None, end_year=None):
    """Get total number of articles for a search without scraping all"""
    try:
        # Get first page to find total articles and pages
        first_page_url = build_pubmed_url(keyword, time_filter, start_year, end_year, page=1)
        response = requests.get(first_page_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get total articles from results-amount div
        total_articles = 0
        results_amount_div = soup.select_one('div.results-amount')
        if results_amount_div:
            value_span = results_amount_div.select_one('span.value')
            if value_span:
                # Remove commas and convert to int
                articles_text = value_span.text.strip().replace(',', '')
                total_articles = int(articles_text)
        
        # Get total pages from pagination
        total_pages = get_total_pages(first_page_url)
        
        # Check if we hit PubMed's 1000-page limit
        scrapable_articles = total_articles
        scrapable_pages = total_pages
        limited_by_pubmed = total_pages > 1000
        
        if limited_by_pubmed:
            # PubMed limits to 1000 pages max (10,000 articles scrapable)
            scrapable_pages = 1000
            scrapable_articles = min(10000, total_articles)  # Can't scrape more than 10,000
        
        return total_articles, scrapable_articles, total_pages, scrapable_pages, limited_by_pubmed
            
    except Exception as e:
        print(f"Error getting article count: {e}")
        return 0, 0, 0, 0, False

def extract_article_info(article_url):
    """Extract title, authors, abstract, journal, and year from an article page"""
    try:
        response = requests.get(article_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title
        title_element = soup.select_one('h1.heading-title')
        title = title_element.text.strip() if title_element else "Title not found"
        
        # Extract authors
        authors = []
        authors_container = soup.select_one('div.inline-authors div.authors')
        if authors_container:
            authors_list = authors_container.select_one('div.authors-list')
            if authors_list:
                for author in authors_list.select('span.authors-list-item'):
                    author_name = author.select_one('a.full-name')
                    if author_name:
                        authors.append(author_name.text.strip())
        
        # Extract journal name
        journal_name = "Journal not found"
        journal_button = soup.select_one('button#full-view-journal-trigger')
        if journal_button and journal_button.get('title'):
            journal_name = journal_button['title'].strip()
        
        # Extract publication year
        pub_year = "Year not found"
        cit_span = soup.select_one('span.cit')
        if cit_span:
            cit_text = cit_span.text.strip()
            # Extract everything before first : or ;
            year_part = re.split(r'[:|;]', cit_text)[0].strip()
            # Extract 4-digit year from the text
            year_match = re.search(r'(19|20)\d{2}', year_part)
            if year_match:
                pub_year = year_match.group(0)
        
        # Extract abstract with structured handling
        abstract_text = ""
        abstract_div = soup.select_one('div.abstract-content#eng-abstract')
        
        if abstract_div:
            paragraphs = abstract_div.select('p')
            formatted_paragraphs = []
            
            for p in paragraphs:
                p_text = ""
                
                # Check if paragraph has a subtitle/heading
                subtitle = p.select_one('strong.sub-title')
                if subtitle:
                    p_text += subtitle.text.strip() + " "
                    subtitle.extract()
                
                # Process the remaining text in the paragraph
                for element in p.contents:
                    if element.name == 'sup':
                        p_text += element.text.strip()
                    elif element.string:
                        p_text += element.string.strip()
                
                # Clean up any extra whitespace
                p_text = re.sub(r'\s+', ' ', p_text).strip()
                if p_text:
                    formatted_paragraphs.append(p_text)
            
            if formatted_paragraphs:
                abstract_text = "\n\n".join(formatted_paragraphs)
        
        if not abstract_text:
            abstract_text = "Abstract not found"
        
        return {
            'title': title,
            'authors': ', '.join(authors) if authors else "Authors not found",
            'journal': journal_name,
            'year': pub_year,
            'abstract': abstract_text,
            'url': article_url
        }
    except Exception as e:
        print(f"Error extracting article info from {article_url}: {e}")
        return None

def create_filename_from_search(keyword, time_filter, start_year=None, end_year=None):
    """Create a standardized filename from search parameters"""
    # Clean keyword for filename
    clean_keyword = re.sub(r'[^\w\s-]', '', keyword.lower())
    clean_keyword = re.sub(r'\s+', '_', clean_keyword)
    
    # Create time filter part
    if time_filter == "1yr":
        time_part = "last_1_year"
    elif time_filter == "5yr":
        time_part = "last_5_years"
    elif time_filter == "10yr":
        time_part = "last_10_years"
    elif time_filter == "custom" and start_year and end_year:
        time_part = f"from_{start_year}_to_{end_year}"
    else:
        time_part = "all_time"
    
    # Add current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    return f"{clean_keyword}_{time_part}_as_of_{current_date}"

def generate_embeddings_batch(articles_data, batch_size=32, progress_container=None):
    """Generate PubMedBERT embeddings for articles with detailed progress tracking"""
    
    # Initialize PubMedBERT model
    progress_container.info("🔄 Loading PubMedBERT model...")
    try:
        # Use the specific PubMedBERT model
        model = SentenceTransformer('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
        progress_container.success("✅ PubMedBERT model loaded successfully!")
    except Exception as e:
        progress_container.error(f"❌ Error loading model: {e}")
        return None
    
    total_articles = len(articles_data)
    embeddings_list = []
    
    # Process in batches
    for batch_start in range(0, total_articles, batch_size):
        batch_end = min(batch_start + batch_size, total_articles)
        batch_num = (batch_start // batch_size) + 1
        total_batches = (total_articles + batch_size - 1) // batch_size
        
        # Prepare batch data
        batch_articles = articles_data[batch_start:batch_end]
        batch_texts = []
        
        for article in batch_articles:
            # Combine title and abstract for embedding
            full_text = f"{article['title']}. {article['abstract']}"
            batch_texts.append(full_text)
        
        # Update progress
        progress_container.info(
            f"🔄 Processing embeddings: {batch_end}/{total_articles} articles (Batch {batch_num}/{total_batches})\n"
            f"Current batch: {len(batch_texts)} articles"
        )
        
        try:
            # Generate embeddings for this batch
            batch_embeddings = model.encode(batch_texts, convert_to_numpy=True)
            embeddings_list.extend(batch_embeddings)
            
            # Show current article being processed
            if batch_articles:
                current_title = batch_articles[-1]['title'][:60] + "..." if len(batch_articles[-1]['title']) > 60 else batch_articles[-1]['title']
                progress_container.success(f"✅ Batch {batch_num} complete. Last article: '{current_title}'")
        
        except Exception as e:
            progress_container.error(f"❌ Error processing batch {batch_num}: {e}")
            continue
        
        # Small delay to prevent overwhelming the system
        time.sleep(0.1)
    
    return np.array(embeddings_list)

def create_raw_text_content(articles_data, search_info):
    """Create formatted raw text content for download"""
    content_lines = []
    
    # Header
    content_lines.append("="*80)
    content_lines.append("PUBMED SEARCH RESULTS")
    content_lines.append("="*80)
    content_lines.append(f"Search Query: {search_info['keyword']}")
    content_lines.append(f"Time Filter: {search_info['time_filter']}")
    if search_info.get('start_year') and search_info.get('end_year'):
        content_lines.append(f"Year Range: {search_info['start_year']}-{search_info['end_year']}")
    content_lines.append(f"Total Articles: {len(articles_data)}")
    content_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    content_lines.append("="*80)
    content_lines.append("")
    
    # Articles
    for i, article in enumerate(articles_data, 1):
        content_lines.append("*" * 50)
        content_lines.append(f"ARTICLE {i}/{len(articles_data)}")
        content_lines.append("*" * 50)
        content_lines.append("")
        content_lines.append(f"TITLE: {article['title']}")
        content_lines.append("")
        content_lines.append(f"AUTHORS: {article['authors']}")
        content_lines.append("")
        content_lines.append(f"JOURNAL: {article['journal']}")
        content_lines.append("")
        content_lines.append(f"YEAR: {article['year']}")
        content_lines.append("")
        content_lines.append("ABSTRACT:")
        content_lines.append(article['abstract'])
        content_lines.append("")
        content_lines.append(f"URL: {article['url']}")
        content_lines.append("")
    
    return "\n".join(content_lines)

def scrape_and_embed_articles(keyword, time_filter, start_year=None, end_year=None, progress_container=None):
    """Main function that scrapes articles and generates embeddings"""
    
    # Create search info for filename and metadata
    search_info = {
        'keyword': keyword,
        'time_filter': time_filter,
        'start_year': start_year,
        'end_year': end_year,
        'search_date': datetime.now().strftime('%Y-%m-%d')
    }
    
    progress_container.info("🚀 Starting PubMed scraping...")
    
    # Build initial URL to get total pages
    initial_url = build_pubmed_url(keyword, time_filter, start_year, end_year, page=1)
    
    # Get total pages
    total_pages_found = get_total_pages(initial_url)
    
    # Apply PubMed's 1000-page limit
    if total_pages_found > 1000:
        total_pages = 1000
        progress_container.warning(f"⚠️ PubMed found {total_pages_found:,} pages, limiting to 1000 pages (PubMed restriction)")
    else:
        total_pages = total_pages_found
    
    articles_data = []
    
    # Process each page
    for page in range(1, total_pages + 1):
        page_url = build_pubmed_url(keyword, time_filter, start_year, end_year, page)
        
        progress_container.info(f"📄 Processing page {page}/{total_pages}")
        
        # Get article URLs from this page
        article_urls = get_article_urls(page_url)
        
        # Process each article on this page
        for i, article_url in enumerate(article_urls):
            current_article = len(articles_data) + 1
            total_expected = total_pages * 10  # Rough estimate
            
            progress_container.info(f"📖 Extracting article {current_article} (Page {page}, Article {i+1}/{len(article_urls)})")
            
            # Extract article information
            article_info = extract_article_info(article_url)
            
            if article_info:
                articles_data.append(article_info)
                
                # Show current article
                title_preview = article_info['title'][:50] + "..." if len(article_info['title']) > 50 else article_info['title']
                progress_container.success(f"✅ Extracted: '{title_preview}'")
            else:
                progress_container.error(f"❌ Failed to extract: {article_url}")
            
            # Rate limiting
            time.sleep(0.15)
        
        # Delay between pages
        time.sleep(0.15)
    
    progress_container.success(f"✅ Scraping completed! Total articles: {len(articles_data)}")
    
    # Generate embeddings
    progress_container.info("🧠 Starting embedding generation...")
    embeddings = generate_embeddings_batch(articles_data, progress_container=progress_container)
    
    if embeddings is None:
        progress_container.error("❌ Failed to generate embeddings")
        return None, None
    
    # Create DataFrame with embeddings
    df_data = []
    for i, article in enumerate(articles_data):
        df_data.append({
            'title': article['title'],
            'authors': article['authors'],
            'journal': article['journal'],
            'year': article['year'],
            'abstract': article['abstract'],
            'url': article['url'],
            'search_query': keyword,
            'search_date': search_info['search_date'],
            'embedding': embeddings[i]
        })
    
    embeddings_df = pd.DataFrame(df_data)
    
    # Create raw text content
    raw_text_content = create_raw_text_content(articles_data, search_info)
    
    # Create filename
    filename_base = create_filename_from_search(keyword, time_filter, start_year, end_year)
    
    progress_container.success("✅ All processing completed!")
    
    return {
        'embeddings_df': embeddings_df,
        'raw_text': raw_text_content,
        'filename_base': filename_base,
        'search_info': search_info,
        'total_articles': len(articles_data)
    }

def pubmed_embeddings_page():
    """Main Streamlit interface for PubMed scraping with embeddings"""

    # Initialize session state
    if 'scraping_results' not in st.session_state:
        st.session_state.scraping_results = None
    if 'embedding_results' not in st.session_state:
        st.session_state.embedding_results = None

    st.title("🔬 PubMed to Embeddings")
    st.markdown("---")
    
    st.markdown("""
    ### PubMed Article Scraper with PubMedBERT Embeddings
    Search PubMed articles and generate embeddings for research analysis.
    """)
    
    # Keyword input - clear results when keyword changes
    keyword = st.text_input(
        "Search Keywords:",
        placeholder="e.g., pet image reconstruction, kinetic modeling",
        help="Enter keywords to search in PubMed",
        key="search_keyword"
    )
    
    # Clear previous results when keyword changes
    if 'previous_keyword' not in st.session_state:
        st.session_state.previous_keyword = ""
    
    if keyword != st.session_state.previous_keyword:
        st.session_state.scraping_results = None
        st.session_state.previous_keyword = keyword
    
    # Time filter options - clear results when time filter changes
    time_filter = st.radio(
        "Time Range:",
        options=["all", "1yr", "5yr", "10yr", "custom"],
        format_func=lambda x: {
            "all": "All time",
            "1yr": "Last 1 year", 
            "5yr": "Last 5 years",
            "10yr": "Last 10 years",
            "custom": "Custom range"
        }[x],
        index=0,
        key="time_filter_radio"
    )
    
    # Clear previous results when time filter changes
    if 'previous_time_filter' not in st.session_state:
        st.session_state.previous_time_filter = "all"
    
    if time_filter != st.session_state.previous_time_filter:
        st.session_state.scraping_results = None
        st.session_state.previous_time_filter = time_filter
    
    # Custom year inputs
    start_year = None
    end_year = None
    custom_years_valid = True
    
    if time_filter == "custom":
        st.markdown("**Custom Year Range:**")
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.number_input(
                "Start Year:",
                min_value=1900,
                max_value=datetime.now().year,
                value=2020,
                step=1
            )
        with col2:
            end_year = st.number_input(
                "End Year:",
                min_value=1900,
                max_value=datetime.now().year,
                value=datetime.now().year,
                step=1
            )
        
        if start_year and end_year:
            if start_year > end_year:
                st.error("Start year must be less than or equal to end year")
                custom_years_valid = False
            elif start_year > datetime.now().year or end_year > datetime.now().year:
                st.error("Years cannot be in the future")
                custom_years_valid = False
    
    # Submit button validation
    submit_disabled = False
    submit_help = ""
    
    if not keyword or not keyword.strip():
        submit_disabled = True
        submit_help = "Please enter search keywords"
    elif time_filter == "custom":
        if not start_year or not end_year:
            submit_disabled = True
            submit_help = "Please fill in both start and end years"
        elif not custom_years_valid:
            submit_disabled = True
            submit_help = "Please fix the year range errors"
    
    # Display search preview
    if keyword and keyword.strip():
        st.markdown("### Current Search Parameters:")
        st.write(f"**Keywords:** {keyword}")
        if time_filter != "all":
            if time_filter == "custom":
                if start_year and end_year:
                    st.write(f"**Time Range:** {start_year} - {end_year}")
            else:
                time_labels = {"1yr": "Last 1 year", "5yr": "Last 5 years", "10yr": "Last 10 years"}
                st.write(f"**Time Range:** {time_labels[time_filter]}")
        else:
            st.write(f"**Time Range:** All time")
        
        # Show preview URL - ALWAYS DISPLAY
        preview_url = build_pubmed_url(keyword, time_filter, start_year, end_year)
        st.write(f"**Search URL:** {preview_url}")
        
        # Show preview filename
        if time_filter != "custom" or (start_year and end_year and custom_years_valid):
            preview_filename = create_filename_from_search(keyword.strip(), time_filter, start_year, end_year)
            st.write(f"**Files will be saved as:** `{preview_filename}.pkl` and `{preview_filename}.txt`")
        
        # Get article count
        if time_filter != "custom" or (start_year and end_year and custom_years_valid):
            with st.spinner("🔍 Checking article count..."):
                total_articles, scrapable_articles, total_pages, scrapable_pages, is_limited = get_total_articles_count(keyword.strip(), time_filter, start_year, end_year)
                
                if total_articles > 0:
                    if is_limited:
                        st.warning(f"⚠️ **PubMed Limitation**: {total_articles:,} articles found, but only {scrapable_articles:,} articles scrapable (first {scrapable_pages:,} pages)")
                    else:
                        st.success(f"📊 **Found {total_articles:,} articles** across **{total_pages:,} pages**")
                    
                    # Estimated time
                    estimated_time_minutes = (scrapable_articles * 1.8) / 60  # Including embedding time
                    if estimated_time_minutes < 1:
                        st.info(f"⏱️ **Estimated processing time:** ~{estimated_time_minutes*60:.0f} seconds")
                    else:
                        st.info(f"⏱️ **Estimated processing time:** ~{estimated_time_minutes:.1f} minutes")
                else:
                    st.warning("❌ No articles found for this search.")
    
    # Submit button
    submitted = st.button(
        "🚀 Start Scraping & Generate Embeddings",
        disabled=submit_disabled,
        help=submit_help if submit_disabled else "Click to start scraping and embedding generation",
        use_container_width=True
    )
    
    # Progress container
    progress_container = st.empty()
    
    # Handle form submission
    if submitted and keyword.strip():
        
        # Clear previous results immediately and set processing flag
        st.session_state.scraping_results = None
        st.session_state.currently_processing = True
        
        # Run scraping and embedding
        with st.spinner("Processing..."):
            results = scrape_and_embed_articles(
                keyword.strip(), 
                time_filter, 
                start_year, 
                end_year,
                progress_container
            )
            
            # Clear processing flag
            st.session_state.currently_processing = False
            
            if results:
                st.session_state.scraping_results = results
                st.balloons()  # Add celebration
                st.success("🎉 **Processing completed successfully!**")
                st.info("📥 **Scroll down to download your files!**")
                
                # Show summary
                st.markdown("### Processing Summary:")
                st.write(f"**Articles processed:** {results['total_articles']}")
                st.write(f"**Embeddings generated:** {len(results['embeddings_df'])} (768-dimensional)")
                st.write(f"**Model used:** PubMedBERT")
            else:
                st.error("❌ Processing failed. Please try again.")
    
    # Download section
    if st.session_state.scraping_results:
        st.markdown("---")
        st.markdown("### 📥 Download Files")
        
        results = st.session_state.scraping_results
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download embeddings file
            embeddings_buffer = io.BytesIO()
            pickle.dump(results['embeddings_df'], embeddings_buffer)
            embeddings_buffer.seek(0)
            
            st.download_button(
                label="📊 Download Embeddings (.pkl)",
                data=embeddings_buffer.getvalue(),
                file_name=f"{results['filename_base']}.pkl",
                mime="application/octet-stream",
                help="Pandas DataFrame with articles and PubMedBERT embeddings"
            )
        
        with col2:
            # Download raw text file
            st.download_button(
                label="📄 Download Raw Text (.txt)",
                data=results['raw_text'],
                file_name=f"{results['filename_base']}.txt",
                mime="text/plain",
                help="Formatted text file with all article content"
            )
        
        # Show file info
        st.markdown("### File Information:")
        st.write(f"**Embeddings file:** Contains {len(results['embeddings_df'])} articles with metadata and 768-dimensional PubMedBERT embeddings")
        st.write(f"**Raw text file:** Contains formatted article text for {results['total_articles']} articles")
        st.write(f"**Ready for:** Jupyter notebook analysis, UMAP visualization, temporal comparison")
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    ### Instructions:
    1. **Enter search keywords** and select time range
    2. **Review article count** and estimated processing time  
    3. **Click 'Start Scraping'** to begin processing
    4. **Download both files** when complete
    5. **Import files in Jupyter** for analysis
    
    ### What You Get:
    - **Embeddings file (.pkl)**: DataFrame with articles + 768D PubMedBERT embeddings
    - **Raw text file (.txt)**: Formatted article content for reference
    - **Ready for analysis**: UMAP, t-SNE, temporal comparison, topic clustering
    
    ### Next Steps:
    - Repeat with different keywords/time ranges
    - Use files in Jupyter for manifold analysis
    - Compare topics or temporal evolution
    """)
    
    # Return to login button
    st.markdown("---")
    if st.button("🔙 Return to Login", use_container_width=True):
        st.session_state.current_page = 'login'
        st.rerun()