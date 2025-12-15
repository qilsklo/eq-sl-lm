import requests
import logging

# Suppress pypdf warnings
logging.getLogger("pypdf").setLevel(logging.ERROR)
from bs4 import BeautifulSoup, Tag
import asyncio
import aiohttp
import re
from urllib.parse import urlparse
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType, model
import tiktoken
from bs4 import NavigableString
import io
import pdfprocessor
import os
import glob
import pypdf
from collections import deque


import datetime

MONTH_FEED = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_month.geojson"
HOUR_FEED = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_hour.geojson"

# Collection Names
COLLECTION_PDF = "myshake_pdf"
COLLECTION_WEB = "myshake_web"
processed_urls_collection = "processed_urls"

client = MilvusClient("myshake.db")
embedding_fn = model.DefaultEmbeddingFunction()  # sentence-transformers/all-MiniLM-L6-v2 - 256 token max
max_tokens = 450

def scrape(start_urls):
    if isinstance(start_urls, str):
        start_urls = [start_urls]
        
    queue = deque()
    
    # 1. Load pending URLs from DB
    pending_urls = get_pending_urls()
    print(f"Loaded {len(pending_urls)} pending URLs from database.")
    for url in pending_urls:
        queue.append(url) # Pending URLs from DB are assumed to be normalized when inserted
        
    # 2. Add start_urls if not already in DB
    for url in start_urls:
        norm_url = normalize_url(url)
        if not is_url_known(norm_url):
            add_url_to_db(norm_url, status=0) # 0 = Pending
            queue.append(norm_url)
    
    print(f"Starting scrape with {len(queue)} items in queue.")

    while queue:
        url = queue.popleft()
        # url from queue should already be normalized
        
        # Double check status
        if is_url_processed(url):
            print(f"Skipping already processed URL: {url}")
            continue

        if not is_valid_scheme(url):
            mark_url_processed(url) 
            continue

        print(f"Processing: {url}")
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                print(f"Failed to fetch {url}: Status {response.status_code}")
                mark_url_processed(url) 
                continue
        except Exception as e:
            print(f"Failed to fetch {url} at time [{str(datetime.datetime.now())}]: {e}")
            mark_url_processed(url)
            continue

        if ".pdf" in url:
            process_scrape_pdf(response)
            mark_url_processed(url)
            continue
            
        soup = BeautifulSoup(response.text, 'html.parser')
        process_scrape_html(soup, url)
        
        # Extract links
        current_domain = extract_domain(url)
        links = soup.select('a')
        for x in links:
            href = x.get('href')
            if href is None:
                continue
            
            if isinstance(href, bytes):
                try:
                    href = href.decode('utf-8')
                except UnicodeDecodeError:
                    continue
            
            if href.startswith('/'):
                parsed_uri = urlparse(url)
                base = '{uri.scheme}://{uri.netloc}'.format(uri=parsed_uri)
                href = base + href
            elif not href.startswith('http'):
                if not href.startswith('mailto:') and not href.startswith('tel:'):
                     href = url.rsplit('/', 1)[0] + '/' + href

            # Normalize the extracted link
            norm_href = normalize_url(href)
            
            if extract_domain(norm_href) == current_domain:
                if not is_url_known(norm_href):
                    add_url_to_db(norm_href, status=0) # Add as pending
                    queue.append(norm_href)
        
        # Mark current URL as processed
        mark_url_processed(url)

def normalize_url(url):
    try:
        parsed = urlparse(url)
        # Lowercase scheme and netloc
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path
        
        # Remove trailing slash from path
        if path and path.endswith('/'):
            path = path[:-1]
            
        # Reconstruct without fragment
        # scheme://netloc/path;parameters?query
        # We ignore params for now as they are rarely used in this context
        # We keep query params as they might be significant
        
        normalized = f"{scheme}://{netloc}{path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
            
        return normalized
    except:
        return url

def is_url_known(url):
    res = client.query(
        collection_name=processed_urls_collection,
        filter=f'url == "{url}"',
        output_fields=["url"]
    )
    return len(res) > 0

def is_url_processed(url):
    res = client.query(
        collection_name=processed_urls_collection,
        filter=f'url == "{url}" and status == 1',
        output_fields=["url"]
    )
    return len(res) > 0

def get_pending_urls():
    # Fetch all URLs with status 0
    # Note: Milvus query limit might need pagination for large sets, 
    # but for now let's grab a reasonable batch.
    res = client.query(
        collection_name=processed_urls_collection,
        filter='status == 0',
        output_fields=["url"],
        limit=10000 # Cap for now
    )
    return [r["url"] for r in res]

def add_url_to_db(url, status=0):
    client.insert(
        collection_name=processed_urls_collection,
        data=[{"url": url, "status": status}]
    )

def mark_url_processed(url):
    # Milvus doesn't support update/partial update easily in all versions.
    # We typically delete and re-insert, or use upsert if available.
    # MilvusClient.upsert() is available in newer SDKs.
    client.upsert(
        collection_name=processed_urls_collection,
        data=[{"url": url, "status": 1}]
    )

def process_scrape_pdf(resp):
    reader = pypdf.PdfReader(io.BytesIO(resp.content))
    # Pass metadata if available, but for scraped PDFs we might only have URL
    # We can try to get title from PDF metadata
    title = ""
    author = ""
    try:
        if reader.metadata:
            title = reader.metadata.get('/Title', "")
            author = reader.metadata.get('/Author', "")
    except:
        pass
        
    if not title:
        title = os.path.basename(resp.url)

    chunks = pdfprocessor.chunk_pdf(reader, resp.url, embedding_fn.tokenizer, max_tokens, title=title, author=author)
    if chunks: 
        db_store(chunks, COLLECTION_PDF)
    print(f"Processed PDF; URL: {resp.url}.")

def process_local_pdf(filepath):
    try:
        # Create a file URI for the origin
        file_uri = f"file://{os.path.abspath(filepath)}"
        
        # Deduplication check
        if is_url_processed(file_uri):
            print(f"Skipping already processed local PDF: {filepath}")
            return

        reader = pypdf.PdfReader(filepath)
        
        title = ""
        author = ""
        try:
            if reader.metadata:
                title = reader.metadata.get('/Title', "")
                author = reader.metadata.get('/Author', "")
        except:
            pass
            
        if not title:
            title = os.path.basename(filepath)

        chunks = pdfprocessor.chunk_pdf(reader, file_uri, embedding_fn.tokenizer, max_tokens, title=title, author=author)
        if chunks:
            db_store(chunks, COLLECTION_PDF)
            # Mark as processed only after successful store
            mark_url_processed(file_uri)
            print(f"Processed local PDF: {filepath}")
        else:
            print(f"No chunks extracted from {filepath}")
            
    except Exception as e:
        print(f"Failed to process local PDF {filepath}: {e}")



def extract_domain(url):
    return urlparse(url).netloc
def is_valid_scheme(url):
    try:
        x = urlparse(url).scheme
        return "http" in x
    except:
        return False
        
def process_scrape_html(soup, url):
    
    chunks = chunk_soup(soup, url) # an array of dicts
    if chunks: 
        db_store(chunks, COLLECTION_WEB)
    print(f"Visited page; URL: {url}. Inserted: {bool(chunks)}")

def chunk_soup(soup, url):
    """
    Improved chunking logic for RAG:
    1. Prune navigational/irrelevant elements.
    2. Group content by heading (semantic chunking).
    3. Enforce a minimum token length for paragraphs by merging short, adjacent blocks.
    4. Handle max token length by splitting by sentence (with overlap) or list item.
    """
    
    tokenizer = embedding_fn.tokenizer
    MIN_TOKENS = 15  # New: Minimum token count for a valid content chunk
    OVERLAP_SENTENCES = 2 # Overlap for paragraph splitting
    OVERLAP_LIST_ITEMS = 3 # Overlap for list splitting
    
    # --- Helper Functions (Minor adjustments for robustness) ---
    
    def count_tokens(text):
        return len(tokenizer.encode(text))

    def split_paragraph(text, heading):
        # Improved: Use OVERLAP_SENTENCES constant for look-back
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []
        curr = []
        for sent in sentences:
            curr.append(sent)
            joined = heading + "\n" + " ".join(curr)
            if count_tokens(joined) > max_tokens:
                # If adding the new sentence makes it too long, package the previous content
                # Overlap: Take the last OVERLAP_SENTENCES sentences as context for the next chunk
                overlap_size = min(len(curr) - 1, OVERLAP_SENTENCES)
                
                prev_content = " ".join(curr[:-1]) # Content up to the sentence that made it too long
                
                # Check minimum length for the chunk being created
                if count_tokens(prev_content) >= MIN_TOKENS:
                     chunks.append(heading + "\n" + prev_content)
                
                # Start the next chunk with the overlap
                curr = curr[-overlap_size:] # Start next chunk with the overlapping sentences
                curr.append(sent) # Add the current sentence
        
        # Add the final buffer content if it meets min length
        final_content = " ".join(curr)
        if final_content and count_tokens(final_content) >= MIN_TOKENS:
            chunks.append(heading + "\n" + final_content)
        return chunks

    def split_list(items, heading, is_ordered):
        # Improved: Use OVERLAP_LIST_ITEMS constant for look-back
        chunks = []
        curr = []
        for i, li in enumerate(items, start=1):
            marker = f"{i}. " if is_ordered else "- "
            curr.append(marker + li)
            joined = heading + "\n" + "\n".join(curr)
            
            if count_tokens(joined) > max_tokens:
                # If adding the new item makes it too long, package the previous content
                overlap_size = min(len(curr) - 1, OVERLAP_LIST_ITEMS)
                
                prev_content = "\n".join(curr[:-1]) # Content up to the list item that made it too long
                
                # Check minimum length for the chunk being created
                if count_tokens(prev_content) >= MIN_TOKENS:
                    chunks.append(heading + "\n" + prev_content)
                
                # Start the next chunk with the overlap
                curr = curr[-overlap_size:] 
                curr.append(marker + li) # Add the current item
                
        # Add the final buffer content
        final_content = "\n".join(curr)
        if final_content and count_tokens(final_content) >= MIN_TOKENS:
            chunks.append(heading + "\n" + final_content)
        return chunks

    def serialize_html(elements):
        return "".join(str(e) for e in elements)
    
    # --- Pruning: Remove irrelevant nodes before processing ---
    
    # List of common tags/classes for navigation, ads, headers, and footers
    irrelevant_selectors = [
        'header', 'footer', 'nav', '.sidebar', '.ad', '.ads',
        '#menu', '#navigation', '.skip-link',
    ]
    
    for selector in irrelevant_selectors:
        for element in soup.select(selector):
            element.decompose() # Remove the element from the soup
            
    # --- Main Chunker Logic ---

    all_chunks = []
    current_heading = "Untitled Section"
    buffer = []
    buffer_html = []
    
    # Now, find all content nodes after pruning
    node_list = soup.find_all(["h1","h2","h3","h4","h5","h6","p","ul","ol"])
    
    # Extract site name (simple approximation)
    site_name = urlparse(url).netloc

    def flush_paragraph_buffer():
        nonlocal buffer, buffer_html
        if not buffer:
            return []
            
        text = " ".join(buffer).strip()
        
        # New: Enforce MIN_TOKENS before splitting for max length
        if count_tokens(text) < MIN_TOKENS and count_tokens(text) > 0:
            # If the combined content is still too short, discard it as noise.
            buffer = []
            buffer_html = []
            return []
            
        out = []
        if count_tokens(current_heading + "\n" + text) > max_tokens:
            out = split_paragraph(text, current_heading)
        else:
            out = [current_heading + "\n" + text]
            
        html_snippet = serialize_html(buffer_html)
        buffer = []
        buffer_html = []
        return [(t, html_snippet) for t in out]

    for el in node_list:
        if el.name in ["h1","h2","h3","h4","h5","h6"]:
            # 1. Flush previous content block
            flushed = flush_paragraph_buffer()
            for txt, html_snip in flushed:
                all_chunks.append({
                    "text": txt,
                    "url": url,
                    "heading": current_heading,
                    "site_name": site_name,
                    "crawl_date": datetime.datetime.now(datetime.timezone.utc).isoformat()
                })
            # 2. Update heading for the next block
            current_heading = el.get_text(strip=True) or "Untitled Section"

        elif el.name == "p":
            txt = el.get_text(" ", strip=True)
            # 3. Collect paragraphs in a buffer
            if txt:
                buffer.append(txt)
                buffer_html.append(el)

        elif el.name in ["ul","ol"]:
            # 1. Flush previous paragraph buffer (if any)
            flushed = flush_paragraph_buffer()
            for txt, html_snip in flushed:
                all_chunks.append({
                    "text": txt,
                    "url": url,
                    "heading": current_heading,
                    "site_name": site_name,
                    "crawl_date": datetime.datetime.now(datetime.timezone.utc).isoformat()
                })

            # 2. Process the list (which is now self-contained)
            items = []
            for li in el.find_all("li", recursive=False):
                li_text = li.get_text(" ", strip=True)
                if li_text:
                     items.append(li_text)

            if items:
                is_ordered = el.name == "ol"
                list_chunks = split_list(items, current_heading, is_ordered)
                html_snippet = serialize_html([el])
                
                for chunk_text in list_chunks:
                    all_chunks.append({
                        "text": chunk_text,
                        "url": url,
                        "heading": current_heading,
                        "site_name": site_name,
                        "crawl_date": datetime.datetime.now(datetime.timezone.utc).isoformat()
                    })

    # 3. Final flush for any remaining buffered paragraphs
    flushed = flush_paragraph_buffer()
    for txt, html_snip in flushed:
        all_chunks.append({
            "text": txt,
            "url": url,
            "heading": current_heading,
            "site_name": site_name,
            "crawl_date": datetime.datetime.now(datetime.timezone.utc).isoformat()
        })

    return all_chunks


def db_store(chunks, collection_name):
    if not chunks:
        return
        
    # Extract text for embedding
    docs = [c["text"] for c in chunks]
    
    vectors = embedding_fn.encode_documents(docs)

    for i in range(len(chunks)):
        chunks[i]["vector"] = vectors[i]

    res = client.insert(collection_name=collection_name, data=chunks)
    print(f"Inserted {len(chunks)} chunks into {collection_name}. Res: {res}")

def fetch_earthquake_feed(url):
    print(f"Fetching earthquake feed: {url}")
    try:
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"Failed to fetch feed: {resp.status_code}")
            return []
        
        data = resp.json()
        features = data.get('features', [])
        chunks = []
        
        for f in features:
            props = f['properties']
            # Annotate
            try:
                time_str = datetime.datetime.fromtimestamp(props['time'] / 1000, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                text = f"Magnitude {props['mag']} earthquake at {props['place']} on {time_str}. Status: {props['status']}."
                origin = props['url']
                
                chunks.append({
                    "text": text,
                    "url": origin,
                    "heading": "Earthquake Report",
                    "site_name": "USGS Earthquake Feed",
                    "crawl_date": datetime.datetime.now(datetime.timezone.utc).isoformat()
                })
            except Exception as e:
                print(f"Error processing feature: {e}")
                continue
                 
        return chunks
    except Exception as e:
        print(f"Error fetching feed: {e}")
        return []


def init_collection():
    
    # 1. PDF Collection
    if not client.has_collection(collection_name=COLLECTION_PDF):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="page_num", dtype=DataType.INT64),
            FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="doi", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="publication_year", dtype=DataType.INT64),
        ]

        schema = CollectionSchema(fields, description="PDF Documents")
        client.create_collection(
            collection_name=COLLECTION_PDF,
            schema=schema,
        )

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        client.create_index(collection_name=COLLECTION_PDF, index_params=index_params)
        print(f"Collection '{COLLECTION_PDF}' created.")
    else:
        print(f"Collection '{COLLECTION_PDF}' already exists.")

    # 2. Web Collection
    if not client.has_collection(collection_name=COLLECTION_WEB):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="crawl_date", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="site_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="heading", dtype=DataType.VARCHAR, max_length=512),
        ]

        schema = CollectionSchema(fields, description="Web Scraped Data")
        client.create_collection(
            collection_name=COLLECTION_WEB,
            schema=schema,
        )

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        client.create_index(collection_name=COLLECTION_WEB, index_params=index_params)
        print(f"Collection '{COLLECTION_WEB}' created.")
        
        # Initial load of month feed
        print("Performing initial load of month feed...")
        chunks = fetch_earthquake_feed(MONTH_FEED)
        if chunks:
            db_store(chunks, COLLECTION_WEB)

    else:
        print(f"Collection '{COLLECTION_WEB}' already exists.")

    # 3. Processed URLs Collection (for persistence/deduplication)
    if not client.has_collection(collection_name=processed_urls_collection):
        url_fields = [
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=2048, is_primary=True),
            FieldSchema(name="status", dtype=DataType.INT64), # 0 = Pending, 1 = Processed
        ]
        url_schema = CollectionSchema(url_fields, description="Track processed URLs with status")
        client.create_collection(
            collection_name=processed_urls_collection,
            schema=url_schema
        )
        print(f"Collection '{processed_urls_collection}' created with status field.")
    else:
        print(f"Collection '{processed_urls_collection}' already exists.")


if __name__ == '__main__':

    init_collection()
    
    # #Process local PDFs first
    pdf_files = glob.glob("docs/*.pdf")
    print(f"Found {len(pdf_files)} local PDF files in docs/")
    for pdf_file in pdf_files:
        process_local_pdf(pdf_file)

    u =["https://www.earthquakecountry.org/",
        "https://www.usgs.gov/programs/earthquake-hazards/faqs-category",
        "https://myshake.berkeley.edu",
        "https://seismo.berkeley.edu",
        "https://www.ready.gov/earthquakes",
        "https://www.redcross.org/get-help/how-to-prepare-for-emergencies/types-of-emergencies/earthquake.html",
        "https://www.caloes.ca.gov/",
        "https://www.gdacs.org/",
        "https://www.ifrc.org/earthquake",
        ]
    
    scrape(u)
