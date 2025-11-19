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
from urllib.parse import urlparse
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType, model
from chunking import *

collection_name = "myshake"
client = MilvusClient("myshake.db")
embedding_fn = model.DefaultEmbeddingFunction()

def scrape(pageurl: str):
    memo = set()
    def scr(pu):
        if not is_valid_scheme(pu): 
            #print(f"BAD URL: {pu}")
            return None
        memo.add(pu)
        response = requests.get(pu)
        if ".pdf" in pu:
            process_scrape_pdf(response, pu)
            return
        soup = BeautifulSoup(response.text, 'html.parser')
        link = soup.select('a')
        process_scrape_html(soup, pu)
        if extract_domain(pu) == extract_domain(pageurl): #Recur
            for x in link:
                y = x.get('href')
                if y is None:
                    continue
            
            # Convert to string explicitly if it might be bytes
                if isinstance(y, bytes):
                    try:
                        y = y.decode('utf-8')
                    except UnicodeDecodeError:
                        # Handle cases where decoding fails, maybe skip the link
                        continue
                if y not in memo:
                    scr(y)
    scr(pageurl)

def extract_domain(url):
    return urlparse(url).netloc
def is_valid_scheme(url):
    x = urlparse(url).scheme
    return "http" in x
def process_scrape_html(soup, url):
    
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
    
    chunks = [] # an array of strings. A chunk is a string which started with an html heading
    current_chunk = []
    current_section_title = ""
    elements = soup.select("h1, h2, h3, h4, h5, h6, p, li")

    for el in elements:
        text = el.get_text(" ", strip=True)
        print(text, end="-----------\n\n")
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
    
    return [{"chktext": c, "origin":url} for c in chunks]


def db_store(chunks): # This duplicates info btw


    docs = [c["chktext"] for c in chunks]
    
    vectors = embedding_fn.encode_documents(docs)

    for i in range(len(chunks)):
        chunks[i]["vector"] = vectors[i]

    res = client.insert(collection_name="myshake",data=chunks)
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


if __name__ == '__main__':

    init_collection()
    #u = input("Ente0r URL to scrape recursively (1-level depth): ")
    #u = ["https://www.usgs.gov/programs/earthquake-hazards/faqs-category"]
    u = ["https://www.earthquakecountry.org/"]
    for l in u:
        scrape(l)
