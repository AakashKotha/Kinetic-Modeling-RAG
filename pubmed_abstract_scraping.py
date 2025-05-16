import requests
from bs4 import BeautifulSoup
import time
import csv
import os
import re

def get_article_urls(search_url):
    """Extract article URLs from the search results page."""
    response = requests.get(search_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    article_links = []
    for link in soup.select('a.docsum-title'):
        article_url = "https://pubmed.ncbi.nlm.nih.gov" + link['href']
        article_links.append(article_url)
    
    return article_links

def extract_article_info(article_url):
    """Extract title, authors, and abstract from an article page."""
    response = requests.get(article_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract title (first occurrence)
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
    
    # Extract abstract - improved to handle structured abstracts with sections
    abstract_text = ""
    abstract_div = soup.select_one('div.abstract-content#eng-abstract')
    
    if abstract_div:
        paragraphs = abstract_div.select('p')
        formatted_paragraphs = []
        
        for p in paragraphs:
            # Get the paragraph text while preserving structure
            p_text = ""
            
            # Check if paragraph has a subtitle/heading
            subtitle = p.select_one('strong.sub-title')
            if subtitle:
                p_text += subtitle.text.strip() + " "
                # Remove the subtitle element so it's not duplicated
                subtitle.extract()
            
            # Process the remaining text in the paragraph
            # Handle any superscript tags by converting them inline
            for element in p.contents:
                if element.name == 'sup':
                    p_text += element.text.strip()
                elif element.string:
                    p_text += element.string.strip()
            
            # Clean up any extra whitespace
            p_text = re.sub(r'\s+', ' ', p_text).strip()
            if p_text:
                formatted_paragraphs.append(p_text)
        
        # Join paragraphs with double newlines to maintain separation
        if formatted_paragraphs:
            abstract_text = "\n\n".join(formatted_paragraphs)
    
    if not abstract_text:
        abstract_text = "Abstract not found"
    
    return {
        'Title': title,
        'Authors': ', '.join(authors),  # Join authors as a string
        'Abstract': abstract_text,
        'URL': article_url
    }

def main():
    # Define the base URL and page range
    base_url = "https://pubmed.ncbi.nlm.nih.gov/?term=PET%20tracer%20kinetic%20modeling&page="
    start_page = 1
    end_page = 119  # Adjust this to the last page you want to scrape
    
    # CSV file for saving results
    csv_file = "pubmed_articles.csv"
    
    if os.path.exists(csv_file):
        os.remove(csv_file)  # Remove existing file to avoid appending
    
    # Create CSV file and write header
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Title', 'Authors', 'Abstract', 'URL']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
    
    # Process all pages
    total_articles = 0
    for page in range(start_page, end_page + 1):
        search_url = f"{base_url}{page}"
        print(f"\nProcessing page {page} of {end_page}: {search_url}")
        
        try:
            # Get article URLs from this page
            article_urls = get_article_urls(search_url)
            print(f"Found {len(article_urls)} articles on page {page}.")
            
            # Extract information from each article on this page
            for i, url in enumerate(article_urls):
                total_articles += 1
                print(f"Processing article {i+1}/{len(article_urls)} on page {page} (Total: {total_articles}): {url}")
                try:
                    article_info = extract_article_info(url)
                    
                    # Append to CSV file (open in append mode)
                    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writerow(article_info)
                    
                    # Add a delay to avoid overwhelming the server
                    time.sleep(0.15)
                except Exception as e:
                    print(f"Error processing {url}: {e}")
            
            # Add a delay between pages to be nice to the server
            time.sleep(0.15)
        except Exception as e:
            print(f"Error processing page {page}: {e}")
            # Continue with the next page
            continue
    
    print(f"\nAll pages processed. Total articles: {total_articles}")
    print(f"Results saved to {csv_file}")

if __name__ == "__main__":
    main()