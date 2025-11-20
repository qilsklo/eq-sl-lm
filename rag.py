# RAG = Retrieval Augmented Generation

import scraper
client = scraper.client
embedding_fn = scraper.embedding_fn # 384-dim ONNX embedding, 
collection_name = scraper.collection_name

def answer_query(query):

    query_vectors = embedding_fn.encode_queries([query])

    res = client.search(
        collection_name=collection_name,
        data=query_vectors,
        limit=5,
        output_fields=["chktext", "origin"],
    )

    return res

if __name__ == '__main__':
    while True:
        q = input("Enter your query: ")
        print(answer_query(q))
