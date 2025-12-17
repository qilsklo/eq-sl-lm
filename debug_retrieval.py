
import standardscraper
import llm
from pymilvus import MilvusClient

def debug_retrieval(query):
    print(f"--- Debugging Query: '{query}' ---")
    
    # 1. Check if 1868 page is in DB
    res = standardscraper.client.query(
        collection_name=standardscraper.COLLECTION_WEB,
        filter='url like "https://seismo.berkeley.edu/eqInfo/1868_quake.html%"',
        output_fields=["url", "heading", "text"],
        limit=5
    )
    print(f"\n[Check] Found {len(res)} chunks for 1868_quake.html")
    if res:
        print(f"Sample 1868 chunk: {res[0]['heading']} - {res[0]['text'][:100]}...")

    # 2. Perform Search
    print(f"\n[Search] Running search_knowledge_base('{query}')...")
    results = llm.search_knowledge_base(query, limit=15)
    
    print(f"\n[Results] Retrieved {len(results)} docs:")
    has_1868 = False
    has_1906 = False
    
    for i, doc in enumerate(results):
        content = doc['content']
        title = doc.get('title') or doc.get('heading') or doc.get('site_name')
        print(f"{i+1}. [{doc['type']}] {title} (Score: {doc['score']:.4f})")
        # print(f"   Snippet: {content[:100]}...")
        
        if "1868" in content or "1868" in str(title):
            has_1868 = True
            print("   -> Contains 1868")
        if "1906" in content or "1906" in str(title):
            has_1906 = True
            print("   -> Contains 1906")

    print(f"\nSummary: Has 1868? {has_1868}. Has 1906? {has_1906}.")

if __name__ == "__main__":
    debug_retrieval("1868 quake damage vs 1906")
