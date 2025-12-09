from pymilvus import MilvusClient

client = MilvusClient("myshake.db")
collection_name = "processed_urls"

if client.has_collection(collection_name):
    # Check pending
    res_pending = client.query(collection_name=collection_name, filter='status == 0', output_fields=["count(*)"])
    # Milvus Lite might not support count(*) in query directly for all versions, let's just get IDs or limit
    # Actually, let's just get a list and count in python for now to be safe with Lite
    res_pending = client.query(collection_name=collection_name, filter='status == 0', output_fields=["url"])
    print(f"Pending URLs (status=0): {len(res_pending)}")
    if len(res_pending) > 0:
        print(f"Sample pending: {res_pending[0]}")

    # Check processed
    res_processed = client.query(collection_name=collection_name, filter='status == 1', output_fields=["url"])
    print(f"Processed URLs (status=1): {len(res_processed)}")
    if len(res_processed) > 0:
        print(f"Sample processed: {res_processed[0]}")
        
else:
    print(f"Collection {collection_name} does NOT exist.")
