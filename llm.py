
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

def perform_filtered_search(query, filter_expr, limit=5):
    """
    Performs vector search with a filter.
    """
    query_vectors = standardscraper.embedding_fn.encode_queries([query])
    res = standardscraper.client.search(
        collection_name=standardscraper.collection_name,
        data=query_vectors,
        limit=limit,
        filter=filter_expr,
        output_fields=["chktext"]
    )
    chunks = []
    for hits in res:
        for hit in hits:
            chunks.append(hit['entity']['chktext'])
    return chunks

def generate_search_params(user_query, api_key):
    """
    Uses LLM to parse the query into semantic query and structured filter.
    """
    genai.configure(api_key=api_key)
    # Using a faster model for this utility task
    model = genai.GenerativeModel('gemini-2.0-flash') 
    
    current_datetime = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    prompt = f"""You are an expert RAG query planner. Your task is to analyze the user's request and output a precise JSON object containing the `semantic_query` and a `filter_expression`.

    Context:
    Current Date and Time: {current_datetime}
    
    Database Schema:
    - date_utc (ISO 8601 String, e.g., '2025-12-04 18:30:00 UTC')
    - magnitude (Float)
    - location (String)
    - heading (String)

    Filter Syntax (Python-like):
    - Date (Relative): "last 7 days" -> date_utc > '2025-11-27 ...'
    - Date (Absolute): "in 2024" -> date_utc >= '2024-01-01 ...' AND date_utc <= '2024-12-31 ...'
    - Magnitude: "over mag 6" -> magnitude >= 6.0
    - Location: "in California" -> location like 'California' (Note: Milvus Lite supports 'like' for prefix/suffix matching if enabled, but for now use '==' or simple string comparisons if possible, or assume 'like' works for substrings in this specific implementation context. Actually, Milvus Lite has limited string filtering. Let's stick to standard comparisons or 'like' with wildcards if supported. For this prompt, assume standard SQL-like syntax is converted by the system, but produce 'like "%...%"' for partial matches if needed, or '==' for exact.)
    *Wait, Milvus Lite string filtering is limited.* 
    *Revised Instruction for Location*: Use `like "Pattern%"` or `== "Exact"`. If complex text match is needed, leave it to vector search and don't filter by location unless it's a strict category.
    
    - Location: "Near California" -> location like '%California%' (ALWAYS use wildcards % for location filtering)
    
    Instructions:
    1. Analyze `user_query` for constraints.
    2. Translate to `filter_expression`.
    3. If no structured constraints, `filter_expression` is "".
    4. `semantic_query` is the text part.
    
    Output JSON only:
    {{
      "semantic_query": "...", 
      "filter_expression": "..." 
    }}
    
    User Query: {user_query}
    """
    
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text)
    except Exception as e:
        print(f"Error generating search params: {e}")
        return {"semantic_query": user_query, "filter_expression": ""}

def query_rag(user_query, history, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # 1. Plan the Search
    plan = generate_search_params(user_query, api_key)
    semantic_query = plan.get("semantic_query", user_query)
    filter_expr = plan.get("filter_expression", "")
    
    # print(f"[DEBUG] Plan: {plan}")

    # Format history
    history_text = ""
    for role, msg in history:
        history_text += f"{role}: {msg}\n"
    history_text += f"User: {user_query}"
    
    # Always retrieve both types of context
    context_chunks = []
    
    # 2. Vector Search
    # If we have a filter, use it. Otherwise, standard search.
    if filter_expr:
        # print(f"[DEBUG] Performing Filtered Search: {filter_expr}")
        try:
            filtered_chunks = perform_filtered_search(semantic_query, filter_expr, limit=10)
            for chunk in filtered_chunks:
                context_chunks.append(f"[FILTERED RESULT] {chunk}")
        except Exception as e:
            print(f"[ERROR] Filtered search failed: {e}. Falling back to standard search.")
            search_chunks = perform_vector_search(semantic_query, limit=5)
            for chunk in search_chunks:
                context_chunks.append(f"[SEARCH RESULT] {chunk}")
    else:
        # Standard search (General + Earthquake Reports)
        search_chunks = perform_vector_search(semantic_query, limit=5)
        for chunk in search_chunks:
            context_chunks.append(f"[SEARCH RESULT] {chunk}")
            
        # Also try to get reports specifically if no filter was applied, just in case
        report_chunks = perform_filtered_search(semantic_query, filter_expr='heading == "Earthquake Report"', limit=5)
        for chunk in report_chunks:
             formatted_chunk = f"[SEARCH RESULT] {chunk}"
             if formatted_chunk not in context_chunks:
                context_chunks.append(formatted_chunk)

    # 3. Latest Earthquakes (Always good context)
    recent_eqs = get_recent_earthquakes(limit=3)
    for eq in recent_eqs:
        context_chunks.append(f"[LATEST REPORT] {eq['chktext']}")
        
    context_text = "\n\n".join(context_chunks)
    # print(f"[DEBUG] Context Text:\n{context_text}\n[DEBUG] End Context")
    
    # 4. Generate Answer
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
    2. **If the user asks for a definition, explanation, or concept (e.g., "What are P-waves?", "Explain subduction") OR the retrieved context is factually incomplete (e.g., the context contains no US reports, but the user asks about California), you are fully authorized to use your extensive internal knowledge base to provide a complete and accurate answer.** Do not deny the existence of widely known facts (e.g., that earthquakes occur in California) just because the context is silent. Use your internal knowledge to fill obvious factual gaps and ensure completeness.
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
