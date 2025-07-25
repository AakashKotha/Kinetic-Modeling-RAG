"""
Pubmed to Embeddings Module
Complete implementation for PubMed scraping with Streamlit interface
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
import time
import re
import threading
from datetime import datetime

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

def scrape_pubmed_articles(keyword, time_filter, start_year=None, end_year=None):
    """Main scraping function that processes all pages (up to PubMed's 1000-page limit)"""
    print(f"\n{'='*80}")
    print(f"STARTING PUBMED SCRAPE")
    print(f"Keyword: {keyword}")
    print(f"Time Filter: {time_filter}")
    if time_filter == "custom":
        print(f"Year Range: {start_year}-{end_year}")
    print(f"{'='*80}\n")
    
    # Build initial URL to get total pages
    initial_url = build_pubmed_url(keyword, time_filter, start_year, end_year, page=1)
    print(f"Search URL: {initial_url}")
    
    # Get total pages
    total_pages_found = get_total_pages(initial_url)
    
    # Apply PubMed's 1000-page limit
    if total_pages_found > 1000:
        total_pages = 1000
        print(f"PubMed found {total_pages_found:,} pages, but limiting to 1000 pages (PubMed restriction)")
        print(f"Will scrape maximum 10,000 articles from first 1000 pages")
    else:
        total_pages = total_pages_found
        print(f"Total pages to process: {total_pages}")
    
    print()
    
    total_articles_processed = 0
    
    # Process each page (up to 1000 max)
    for page in range(1, total_pages + 1):
        page_url = build_pubmed_url(keyword, time_filter, start_year, end_year, page)
        print(f"\n{'-'*60}")
        print(f"PROCESSING PAGE {page} of {total_pages}")
        if total_pages_found > 1000:
            print(f"(Note: PubMed has {total_pages_found:,} total pages, scraping first 1000 only)")
        print(f"Page URL: {page_url}")
        print(f"{'-'*60}")
        
        # Get article URLs from this page
        article_urls = get_article_urls(page_url)
        print(f"Found {len(article_urls)} articles on page {page}")
        
        # Process each article on this page
        for i, article_url in enumerate(article_urls):
            total_articles_processed += 1
            print(f"\n{'*'*40}")
            print(f"ARTICLE {i+1}/{len(article_urls)} on PAGE {page}")
            print(f"TOTAL ARTICLES PROCESSED: {total_articles_processed}")
            print(f"{'*'*40}")
            
            # Extract article information
            article_info = extract_article_info(article_url)
            
            if article_info:
                # Print article details to terminal
                print(f"\nTITLE: {article_info['title']}")
                print(f"\nAUTHORS: {article_info['authors']}")
                print(f"\nJOURNAL: {article_info['journal']}")
                print(f"\nYEAR: {article_info['year']}")
                print(f"\nABSTRACT:\n{article_info['abstract']}")
                print(f"\nURL: {article_info['url']}")
            else:
                print(f"Failed to extract information from: {article_url}")
            
            # Rate limiting
            time.sleep(0.15)
        
        # Delay between pages
        time.sleep(0.15)
    
    print(f"\n{'='*80}")
    print(f"SCRAPING COMPLETED")
    print(f"Total articles processed: {total_articles_processed}")
    if total_pages_found > 1000:
        print(f"Note: {total_pages_found - 1000:,} additional pages exist but were not accessible due to PubMed limits")
    print(f"{'='*80}\n")

def pubmed_embeddings_page():
    """Main Streamlit interface for PubMed scraping"""
    st.title("🔬 PubMed to Embeddings")
    st.markdown("---")
    
    st.markdown("""
    ### PubMed Article Scraper
    Search PubMed articles with abstracts and extract detailed information.
    """)
    
    # Keyword input (outside form to make it reactive)
    keyword = st.text_input(
        "Search Keywords:",
        placeholder="e.g., pet image reconstruction, kinetic modeling",
        help="Enter keywords to search in PubMed"
    )
    
    # Time filter options (outside form to make it reactive)
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
        index=0
    )
    
    # Custom year inputs (only show when custom is selected)
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
                step=1,
                key="start_year_input"
            )
        with col2:
            end_year = st.number_input(
                "End Year:",
                min_value=1900,
                max_value=datetime.now().year,
                value=datetime.now().year,
                step=1,
                key="end_year_input"
            )
        
        # Validate custom years
        if start_year and end_year:
            if start_year > end_year:
                st.error("Start year must be less than or equal to end year")
                custom_years_valid = False
            elif start_year > datetime.now().year or end_year > datetime.now().year:
                st.error("Years cannot be in the future")
                custom_years_valid = False
    
    # Submit button logic
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
    
    # Submit button
    submitted = st.button(
        "🔍 Start Scraping",
        disabled=submit_disabled,
        help=submit_help if submit_disabled else "Click to start scraping PubMed articles",
        use_container_width=True
    )
    
    # Display current search parameters and article count
    if keyword and keyword.strip():
        st.markdown("### Current Search Parameters:")
        st.write(f"**Keywords:** {keyword}")
        if time_filter != "all":
            if time_filter == "custom":
                if start_year and end_year:
                    st.write(f"**Time Range:** {start_year} - {end_year}")
                else:
                    st.write(f"**Time Range:** Custom (please fill in years)")
            else:
                time_labels = {"1yr": "Last 1 year", "5yr": "Last 5 years", "10yr": "Last 10 years"}
                st.write(f"**Time Range:** {time_labels[time_filter]}")
        else:
            st.write(f"**Time Range:** All time")
        
        # Show preview URL
        preview_url = build_pubmed_url(keyword, time_filter, start_year, end_year)
        st.write(f"**Search URL:** {preview_url}")
        
        # Get and display article count
        if time_filter != "custom" or (start_year and end_year and custom_years_valid):
            with st.spinner("🔍 Checking article count..."):
                total_articles, scrapable_articles, total_pages, scrapable_pages, is_limited = get_total_articles_count(keyword.strip(), time_filter, start_year, end_year)
                
                if total_articles > 0:
                    if is_limited:
                        # Show limitation warning for searches with >1000 pages
                        st.warning(f"""
                        ⚠️ **PubMed Limitation Detected**
                        
                        **Total articles found:** {total_articles:,} articles across {total_pages:,} pages
                        
                        **Scrapable due to PubMed limits:** {scrapable_articles:,} articles (first {scrapable_pages:,} pages only)
                        
                        PubMed only allows navigation up to page 1000, so {total_articles - scrapable_articles:,} articles are not accessible via scraping.
                        """)
                        
                        st.info(f"📊 **Will scrape {scrapable_articles:,} articles** across **{scrapable_pages:,} pages** (maximum allowed)")
                        
                        # Add estimated time
                        estimated_time_minutes = (scrapable_articles * 0.15) / 60
                        st.write(f"⏱️ **Estimated scraping time:** ~{estimated_time_minutes:.1f} minutes")
                    else:
                        # Normal case - no limitation
                        st.success(f"📊 **Found {total_articles:,} articles** across **{total_pages:,} pages**")
                        
                        # Show breakdown if multiple pages
                        if total_pages > 1:
                            articles_on_last_page = total_articles - (total_pages - 1) * 10
                            if articles_on_last_page > 0:
                                st.info(f"📄 Pages 1-{total_pages-1}: {(total_pages-1) * 10:,} articles (10 per page) | Page {total_pages}: {articles_on_last_page} articles")
                            else:
                                st.info(f"📄 All {total_pages} pages have 10 articles each")
                        
                        # Add estimated time
                        estimated_time_minutes = (total_articles * 0.15) / 60
                        if estimated_time_minutes < 1:
                            st.write(f"⏱️ **Estimated scraping time:** ~{estimated_time_minutes*60:.0f} seconds")
                        else:
                            st.write(f"⏱️ **Estimated scraping time:** ~{estimated_time_minutes:.1f} minutes")
                else:
                    st.warning("❌ No articles found for this search. Try different keywords or time range.")
        else:
            st.info("🔍 Complete the search parameters to see article count")
    
    # Handle form submission
    if submitted and keyword.strip():
        st.success("🚀 Scraping started! Check the Python terminal for detailed output.")
        st.info("⚠️ This process may take a while depending on the number of articles. The detailed article information will be printed to the Python terminal.")
        
        # Add a progress indicator
        with st.spinner("Initializing scraper..."):
            # Start scraping in a separate thread to avoid blocking Streamlit
            def run_scraper():
                scrape_pubmed_articles(keyword.strip(), time_filter, start_year, end_year)
            
            # Run the scraper
            scraper_thread = threading.Thread(target=run_scraper)
            scraper_thread.daemon = True
            scraper_thread.start()
            
            st.success("✅ Scraper is now running! Monitor the Python terminal for progress and results.")
    
    # Instructions
    st.markdown("---")
    st.markdown("""
    ### Instructions:
    1. **Enter search keywords** (e.g., "pet image reconstruction")
    2. **Select time range** or choose custom years
    3. **Click 'Start Scraping'** to begin
    4. **Monitor the Python terminal** for detailed article information
    
    ### Output Format:
    Each article will be printed to the terminal with:
    - **Title**
    - **Authors** 
    - **Abstract** (with structured sections)
    - **URL**
    
    ### Note:
    - Only articles with abstracts are included (filter=simsearch1.fha)
    - Rate limiting is applied to be respectful to PubMed servers
    - Progress is shown in the terminal
    """)
    
    # Return to login button
    st.markdown("---")
    if st.button("🔙 Return to Login", use_container_width=True):
        st.session_state.current_page = 'login'
        st.rerun()