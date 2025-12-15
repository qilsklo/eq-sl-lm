
import os
import datetime
import re
import json
from dotenv import load_dotenv
import google.generativeai as genai
import streamlit as st
import standardscraper

import prompts

# Load environment variables
load_dotenv()

def get_api_key():
    # Try to get from environment variable first
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    
    # Check if running in Streamlit
    try:
        if st.runtime.exists():
            # Try to get from Streamlit secrets
            if "GEMINI_API_KEY" in st.secrets:
                return st.secrets["GEMINI_API_KEY"]

            # If not found, ask user via sidebar
            if "api_key" not in st.session_state:
                st.session_state.api_key = ""
            
            key = st.sidebar.text_input("Enter Gemini API Key", type="password", value=st.session_state.api_key)
            if key:
                st.session_state.api_key = key
                return key
            
            st.warning("Please enter your Gemini API Key in the sidebar to proceed.")
            st.stop()
            return None
    except Exception:
        pass # Not in streamlit or runtime not ready

    # Fallback to CLI input
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
    
    prompt = prompts.SEARCH_PARAM_PROMPT.format(current_datetime=current_datetime, user_query=user_query)
    
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
    recent_eqs = get_recent_earthquakes(limit=10)
    for eq in recent_eqs:
        context_chunks.append(f"[LATEST REPORT] {eq['chktext']}")
        
    context_text = "\n\n".join(context_chunks)
    # print(f"[DEBUG] Context Text:\n{context_text}\n[DEBUG] End Context")
    
    # 4. Generate Answer
    # 3. Generate Answer
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    prompt = prompts.RAG_ANSWER_PROMPT.format(current_datetime=current_datetime, history_text=history_text, context_text=context_text, user_query=user_query)

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling LLM: {e}"

def main_streamlit():
    st.set_page_config(page_title="Earthquake Assistant", page_icon="🌍")
    st.title("🌍 Earthquake Assistant")

    api_key = get_api_key()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask about earthquakes..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Convert Streamlit history to the format expected by query_rag (list of tuples)
                # Capitalize roles to match the expected format in query_rag
                history_tuples = [(msg["role"].capitalize(), msg["content"]) for msg in st.session_state.messages[:-1]]
                
                response = query_rag(prompt, history_tuples, api_key)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def main_cli():
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

if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        if st.runtime.exists():
            main_streamlit()
        else:
            main_cli()
    except Exception:
        main_cli()
