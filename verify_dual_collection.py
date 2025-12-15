
import standardscraper
import llm
import os
import shutil
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType

# Clean up existing DB to start fresh for verification
if os.path.exists("myshake.db"):
    try:
        client = MilvusClient("myshake.db")
        if client.has_collection(standardscraper.COLLECTION_PDF):
            client.drop_collection(standardscraper.COLLECTION_PDF)
        if client.has_collection(standardscraper.COLLECTION_WEB):
            client.drop_collection(standardscraper.COLLECTION_WEB)
        if client.has_collection(standardscraper.processed_urls_collection):
            client.drop_collection(standardscraper.processed_urls_collection)
        print("Dropped existing collections.")
    except Exception as e:
        print(f"Error dropping collections: {e}")

# 1. Initialize Collections (Manually to skip feed fetch)
print("Initializing collections (skipping feed fetch)...")
client = MilvusClient("myshake.db")

# PDF Collection
if not client.has_collection(standardscraper.COLLECTION_PDF):
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
    client.create_collection(collection_name=standardscraper.COLLECTION_PDF, schema=schema)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_index(collection_name=standardscraper.COLLECTION_PDF, index_params=index_params)

# Web Collection
if not client.has_collection(standardscraper.COLLECTION_WEB):
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
    client.create_collection(collection_name=standardscraper.COLLECTION_WEB, schema=schema)
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
    client.create_index(collection_name=standardscraper.COLLECTION_WEB, index_params=index_params)

# Processed URLs
if not client.has_collection(standardscraper.processed_urls_collection):
    url_fields = [
        FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=2048, is_primary=True),
        FieldSchema(name="status", dtype=DataType.INT64),
    ]
    url_schema = CollectionSchema(url_fields, description="Track processed URLs with status")
    client.create_collection(collection_name=standardscraper.processed_urls_collection, schema=url_schema)


# 2. Ingest Dummy Data
print("Ingesting dummy data...")

# Dummy PDF Chunk
pdf_chunk = {
    "text": "This is a test sentence from a PDF document about earthquakes.",
    "title": "Test PDF Document",
    "page_num": 42,
    "author": "Dr. Seismology",
    "doi": "10.1234/test",
    "publication_year": 2023
}
standardscraper.db_store([pdf_chunk], standardscraper.COLLECTION_PDF)

# Dummy Web Chunk
web_chunk = {
    "text": "This is a test sentence from a website about earthquake safety.",
    "url": "https://example.com/safety",
    "site_name": "Example Safety Site",
    "heading": "Drop, Cover, and Hold On",
    "crawl_date": "2023-10-27T10:00:00Z"
}
standardscraper.db_store([web_chunk], standardscraper.COLLECTION_WEB)

# 3. Verify Retrieval
print("Verifying retrieval...")
results = llm.perform_vector_search("earthquake", limit=10)

print("\n--- Retrieval Results ---")
for res in results:
    print(res)
    print("-" * 20)

# Check if we got both types
has_pdf = any("PDF:" in r for r in results)
has_web = any("WEB:" in r for r in results)

if has_pdf and has_web:
    print("\nSUCCESS: Retrieved both PDF and Web documents with correct citations.")
else:
    print(f"\nFAILURE: Missing document types. PDF: {has_pdf}, Web: {has_web}")
