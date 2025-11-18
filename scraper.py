"""
to scrape
https://www.earthquakecountry.org/
https://www.usgs.gov/programs/earthquake-hazards/faqs-category
maybe go one level deep into links on this site
- all headings
- all paragraphs
- all lists (ul, ol) and their list items (li)

"""


import requests
from bs4 import BeautifulSoup
import asyncio
import aiohttp
import re
from urllib.parse import urlparse,urljoin
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType, model

collection_name = "myshake"
client = MilvusClient("myshake.db")
embedding_fn = model.DefaultEmbeddingFunction()

MAX_CONCURRENT_REQUESTS = 5
BATCH_SIZE = 50 # Chunks per Milvus insert
DOMAIN_LIMIT = 1 # Only crawl 1 level deep on the starting domain

async def scrape_async(start_urls):
    """
    Asynchronously crawls URLs, processes content, and yields chunks.
    """
    to_visit = asyncio.Queue()
    visited = set()
    base_domains = {extract_domain(url) for url in start_urls}
    
    # Initialize the queue
    for url in start_urls:
        if is_valid_scheme(url) and url not in visited:
            to_visit.put_nowait(url)
            visited.add(url)
            
    async with aiohttp.ClientSession() as session:
        
        # Worker function to handle fetching and processing
        async def worker():
            nonlocal to_visit, visited
            
            while not to_visit.empty():
                url = await to_visit.get()
                current_domain = extract_domain(url)
                
                print(f"Scraping: {url}")
                
                content, final_url = await fetch(session, url)
                if not content:
                    continue

                soup = BeautifulSoup(content, 'html.parser')
                
                # Yield the scraped data for batch processing
                for chunk in chunk_soup(soup, final_url):
                    yield chunk

                # Find new links andfor chunk in ch add them to the queue
                if current_domain in base_domains:
                    for link_tag in soup.select('a'):
                        href = link_tag.get('href')
                        if href:
                            # Resolve relative URLs
                            new_url = urljoin(final_url, href)
                            
                            # Check domain and scheme validity
                            if extract_domain(new_url) in base_domains and is_valid_scheme(new_url):
                                if new_url not in visited:
                                    visited.add(new_url)
                                    to_visit.put_nowait(new_url)
        
        # Start multiple workers concurrently
        workers = [worker() for _ in range(MAX_CONCURRENT_REQUESTS)]
        
        # Gather the results from the workers' generators
        # This structure allows workers to run non-blockingly while yielding chunks
        for worker_gen in workers:
            async for chunk in worker_gen:
                yield chunk


def extract_domain(url):
    return urlparse(url).netloc
def is_valid_scheme(url):
    x = urlparse(url).scheme
    return "http" in x
def process_scrape(soup, url):

    elements = soup.select("h1, h2, h3, h4, h5, h6, p, li")
    if not elements: return
    chunks = chunk_soup(soup, url) # an array of dicts
    db_success = db_store(chunks)
    print(f"Visited page; soup hash: {hash(soup)}")

def chunk_soup(soup, url):
    """
    Inside a vector DB entry:
    - embedding
    - origin URL
    - title of heading of the chunk
    - original HTML Snippet of the chunk
    1. Take the soup, and create a dict {url, heading, html_snippet}
    2. for each chunk create a dictionary that includes all this info 
    """
    elements = soup.select("h1, h2, h3, h4, h5, h6, p, li") 
    if not elements:
        return []
    chunks = [] # an array of strings. A chunk is a string which started with an html heading
    current_chunk = []
    current_section_title = ""

    for el in elements:
        text = el.get_text(" ", strip=True)

        if el.name.startswith("h"):
            # start a ne0w section

            heading_level = int(el.name[1])

            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
            if heading_level <= 3:
                current_section_title = text
            
            if current_section_title and current_section_title != text:
                current_chunk.append(f"Section: {current_section_title}")
            
            current_chunk.append(text)

        elif el.name == "li":
            current_chunk.append(f"- {text}")

        else:
            current_chunk.append(text)

    # add last chunk
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    #[print(c, end="\n\n") for c in chunks]
    
    chunkst = []
    for i in range(len(chunks)):
        chunkst.append({"chktext":chunks[i],"origin":url})
    chunks = chunkst

    return chunks


def db_store(chunks): # This duplicates info btw


    docs = [c["chktext"] for c in chunks]
    
    vectors = embedding_fn.encode_documents(docs)

    for i in range(len(chunks)):
        chunks[i]["vector"] = vectors[i]

    res = client.insert(collection_name=collection_name,data=chunks)
    print(res)

    # store in vectordb
    # store same chunks in a bm25 index

def init_collection():
    
    if client.has_collection(collection_name=collection_name):
        print("init_collection skipped; collection already exists.")
        return
        #client.drop_collection(collection_name=collection_name)

    fields = [
        # Primary Key field - MUST be set with auto_id=True for automatic ID generation
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        
        # Vector field (Dimension 768)
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
        
        # Scalar fields to store chunk text and origin URL
        # The Milvus Lite (SQLite) backend used with MilvusClient requires scalar fields to be defined.
        FieldSchema(name="chktext", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="origin", dtype=DataType.VARCHAR, max_length=512),
    ]

    schema = CollectionSchema(fields, description="Earthquake website scraped data")
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        # Note: If you want to use Index Types other than FLAT, you'd specify them here.
    )

    index_params = client.prepare_index_params()
    
    # Use AUTOINDEX index for the vector field (because i'm running the db locally, should change to HNSW later)
    index_params.add_index(
        field_name="vector", 
        index_type="AUTOINDEX", # Not a High-Performance Index Type
        metric_type="COSINE" # Metric for distance calculation
    )

    # Apply the index to the collection
    client.create_index(collection_name=collection_name, index_params=index_params)
    print(f"Collection '{collection_name}' created and 'vector' field indexed.")

async def fetch(session, url):
    try:
        async with session.get(url, timeout=10) as response:
            if response.status != 200:
                print(f"Failed to fetch {url} with status: {response.status}")
                return None, None
            content = await response.text()
            final_url = str(response.url)
            return content, final_url
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None, None


if __name__ == '__main__':

    init_collection()
    start_urls = [
        "https://www.earthquakecountry.org/",
        "https://www.usgs.gov/programs/earthquake-hazards/faqs-category"
    ]
    all_chunks = []
    
    # Run the asynchronous scraper and collect chunks
    # Note: We must use a coroutine to iterate over the async generator
    async def main_run():
        global all_chunks
        async for chunk in scrape_async(start_urls):
            all_chunks.append(chunk)
            
            # Check if we've reached the batch size limit
            if len(all_chunks) >= BATCH_SIZE:
                print(f"Batch limit reached ({BATCH_SIZE}). Inserting...")
                db_store(all_chunks)
                all_chunks = [] # Reset batch list

    asyncio.run(main_run())
    
    # Insert any remaining chunks after the scraping is complete
    if all_chunks:
        print(f"Scraping complete. Inserting final batch of {len(all_chunks)} chunks...")
        db_store(all_chunks)

    print("Scraping and ingestion complete.")