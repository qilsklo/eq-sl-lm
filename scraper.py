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
from bs4 import BeautifulSoup, Tag
import asyncio
import aiohttp
import re
from urllib.parse import urlparse
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType, model
import tiktoken
#import pypdf
import re
from bs4 import NavigableString, Tag
import io
#import pdfprocessor


collection_name = "myshake"
client = MilvusClient("myshake.db")
embedding_fn = model.DefaultEmbeddingFunction()  # sentence-transformers/all-MiniLM-L6-v2 - 256 token max
max_tokens = 510

def scrape(pageurl: str):
    memo = set()
    def scr(pu):
        if not is_valid_scheme(pu): 
            #print(f"BAD URL: {pu}")
            return None
        memo.add(pu)
        response = requests.get(pu)
        if ".pdf" in pu:
            process_scrape_pdf(response)
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

def process_scrape_pdf(resp):
    ...
#    reader = pypdf.PdfReader(io.BytesIO(resp.content))
#    chunks = pdfprocessor.chunk_pdf(reader, resp.url)
#    if chunks: db_store(chunks)
#    print(f"Processed PDF; URL: {resp.url}.")



def extract_domain(url):
    return urlparse(url).netloc
def is_valid_scheme(url):
    x = urlparse(url).scheme
    return "http" in x
def process_scrape_html(soup, url):
    
    chunks = chunk_soup(soup, url) # an array of dicts
    if chunks: db_store(chunks)
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
                    "chktext": txt,
                    "origin": url,
                    "heading": current_heading,
                    "html_snippet": html_snip,
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
                    "chktext": txt,
                    "origin": url,
                    "heading": current_heading,
                    "html_snippet": html_snip,
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
                        "chktext": chunk_text,
                        "origin": url,
                        "heading": current_heading,
                        "html_snippet": html_snippet,
                    })

    # 3. Final flush for any remaining buffered paragraphs
    flushed = flush_paragraph_buffer()
    for txt, html_snip in flushed:
        all_chunks.append({
            "chktext": txt,
            "origin": url,
            "heading": current_heading,
            "html_snippet": html_snip,
        })

    return all_chunks

# def chunk_soup(soup, url):
#     """
#     Inside a vector DB entry:
#     - embedding
#     - origin URL
#     - title of heading of the chunk
#     - original HTML Snippet of the chunk
#     1. Take the soup, and create a dict {url, heading, html_snippet}
#     2. for each chunk create a dictionary that includes all this info 
#     """
    
#     chunks = [] # an array of strings.
#     current_chunk = []
#     elements = soup.select("h1, h2, h3, h4, h5, h6, p, li")

#     # some logic --> chunks.append(current_chunk)
#     # some other logic --> current_chunk = [] (reset the current_chunk during creation, probably in a loop)

#     # add last chunk
#     if current_chunk:
#         chunks.append("\n".join(current_chunk))
    
#     return [{"chktext": c, "origin":url} for c in chunks]


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
        FieldSchema(name="heading", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="html_snippet", dtype=DataType.VARCHAR, max_length=512),
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
    #u = []
    u =["https://www.earthquakecountry.org/",
        "https://www.usgs.gov/programs/earthquake-hazards/faqs-category"]
    for l in u:
        scrape(l)
