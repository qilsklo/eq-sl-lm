
import os
import datetime
import re
import json
from dotenv import load_dotenv
import google.generativeai as genai
import standardscraper

# Load environment variables
load_dotenv()

def get_api_key():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print("GEMINI_API_KEY not found in .env")
        key = input("Please enter your Gemini API Key: ").strip()
        if not key:
            print("API Key is required to proceed.")
            exit(1)
    return key

def get_recent_earthquakes(limit=10):
    """
    Fetches recent earthquake reports from the database.
    """
    res = standardscraper.client.query(
        collection_name=standardscraper.collection_name,
        filter='heading == "Earthquake Report"',
        output_fields=["chktext", "origin"],
        limit=100 
    )
    
    if not res:
        return []

    def parse_date(text):
        match = re.search(r'on (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC)', text)
        if match:
            return datetime.datetime.strptime(match.group(1), '%Y-%m-%d %H:%M:%S %Z')
        return datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)

    sorted_res = sorted(res, key=lambda x: parse_date(x['chktext']), reverse=True)
    return sorted_res[:limit]

def perform_vector_search(query, limit=5):
    """
    Performs standard vector search.
    """
    query_vectors = standardscraper.embedding_fn.encode_queries([query])
    res = standardscraper.client.search(
        collection_name=standardscraper.collection_name,
        data=query_vectors,
        limit=limit,
        output_fields=["chktext"]
    )
    chunks = []
    for hits in res:
        for hit in hits:
            chunks.append(hit['entity']['chktext'])
    return chunks

def query_rag(user_query, history, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Format history
    history_text = ""
    for role, msg in history:
        history_text += f"{role}: {msg}\n"
    history_text += f"User: {user_query}"
    
    #print(f"[DEBUG] Retrieving context...")
    
    # Always retrieve both types of context
    context_chunks = []
    
    # 1. Latest Earthquakes (for "recent" queries)
    recent_eqs = get_recent_earthquakes(limit=3)
    for eq in recent_eqs:
        context_chunks.append(f"[LATEST REPORT] {eq['chktext']}")
        
    # 2. Vector Search (for specific/scientific queries)
    search_chunks = perform_vector_search(user_query, limit=3)
    for chunk in search_chunks:
        context_chunks.append(f"[SEARCH RESULT] {chunk}")
        
    context_text = "\n\n".join(context_chunks)
    
    # 3. Generate Answer
    # 3. Generate Answer
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    prompt = f"""You are a helpful assistant that provides information about earthquakes.
    Use the provided context and chat history to answer the user's question.

    --- GROUNDING FACTS ---
    **Current Date and Time: {current_datetime}**
    Use this date for all time-based calculations (e.g., checking if an event was "last week").
    --- GROUNDING FACTS ---

    Context Sources:
    - [LATEST REPORT]: The most recent earthquakes recorded. Use this if the user asks for "latest", "recent", or "last few" earthquakes.
    - [SEARCH RESULT]: Information retrieved based on the user's query. Use this if it matches the user's topic (e.g., specific location, scientific concept).

    Instructions:
    1. If the context contains highly relevant information (especially [SEARCH RESULT] or [LATEST REPORT]), **you must prioritize it** for your answer to ensure accuracy.
    2. **If the user asks for a definition, explanation, or concept (e.g., "What are P-waves?", "Explain subduction") AND the retrieved context is NOT relevant, you are authorized to use your internal knowledge base to provide a thorough answer.** Maintain the persona of an expert.
    3. For time-based questions (e.g., "Was it in the last week?"), use the **Current Date and Time** for accurate calculation against earthquake timestamps.
    4. For follow-up questions (e.g., "Would I have felt it?"), combine the context (earthquake details) with your general knowledge (geography, physics).

    Chat History:
    {history_text}

    Retrieved Context:
    {context_text}

    User: {user_query}
    Assistant:
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling LLM: {e}"

if __name__ == "__main__":
    api_key = get_api_key()
    history = [] # List of (role, message) tuples
    
    print("Earthquake Assistant (Type 'quit' to exit)")
    while True:
        query = input("\nEnter your query: ")
        if query.lower() in ['quit', 'exit']:
            break
        
        answer = query_rag(query, history, api_key)
        print("\nResponse:")
        print(answer)
        
        # Update history
        history.append(("User", query))
        history.append(("Assistant", answer))
        
        # Keep history manageable
        if len(history) > 10:
            history = history[-10:]
