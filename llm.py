
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

import earthquake_data

import earthquake_data

def get_search_params(user_query, api_key):
    """
    Uses LLM to extract search parameters (date, magnitude, etc.) from the query.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    current_time = datetime.datetime.now(datetime.timezone.utc).isoformat()
    prompt = prompts.SEARCH_PARAM_PROMPT.format(
        current_datetime=current_time,
        user_query=user_query
    )
    
    try:
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        params = json.loads(response.text)
        return params
    except Exception as e:
        print(f"Error extracting params: {e}")
        return {}

def perform_vector_search(query, limit=3):
    """
    Performs standard vector search using standardscraper.
    """
    try:
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
    except Exception as e:
        print(f"Vector search failed: {e}")
        return []

def query_rag(user_query, history, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # 1. Fetch latest data (simple polling simulation: fetch on query)
    # In a real production app, this would be a background service.
    try:
        # Fetching all_day to get better context
        new_events = earthquake_data.manager.fetch_feed("all_day")
        earthquake_data.manager.process_features(new_events)
    except Exception as e:
        print(f"Error updating feed: {e}")
        # We continue, but the context will show 'stale' or 'unknown' if last fetch failed/didn't happen recently.

    # 2. Get Search Params
    search_params = get_search_params(user_query, api_key)
    start_date = search_params.get("start_date")
    end_date = search_params.get("end_date")
    semantic_query = search_params.get("semantic_query")
    user_coordinates = search_params.get("user_coordinates")
    
    # Default to UC Berkeley if no location specified
    if not user_coordinates:
        user_coordinates = [37.8715, -122.2730]
        
    if not semantic_query:
        semantic_query = user_query

    # 3. Get Context from EarthquakeManager
    # Extract magnitude constraint
    # First try from LLM params
    min_mag = search_params.get("min_magnitude")
    
    # Fallback to regex if not found in params (and ensure regex is robust)
    if min_mag is None:
        # Regex: Look for 'magnitude', 'mag', 'm' followed by number, OR just 'M' followed by number
        # Use word boundaries to avoid matching 'from' -> 'm'
        mag_match = re.search(r'\b(?:magnitude|mag|m)\s*(\d+(?:\.\d+)?)', user_query, re.IGNORECASE)
        if mag_match:
            try:
                min_mag = float(mag_match.group(1))
            except ValueError:
                pass
    
    # Check if user is asking about a specific event (naive check, or use search params)
    # For now, we just give the latest context.
    context_data = earthquake_data.manager.get_context_for_llm(
        min_magnitude=min_mag,
        start_date=start_date,
        end_date=end_date,
        user_lat=user_coordinates[0] if user_coordinates else None,
        user_lon=user_coordinates[1] if user_coordinates else None
    )
    event_context_json = json.dumps(context_data, indent=2)
    
    # 4. Get Safety Docs via Vector Search
    safety_docs = perform_vector_search(semantic_query, limit=3)
    safety_context_text = "\n\n".join([f"[DOC {i+1}] {doc}" for i, doc in enumerate(safety_docs)])

    # Combine Contexts
    full_context = f"""
--- EVENT DATA (Authoritative) ---
{event_context_json}

--- SAFETY DOCS (Reference) ---
{safety_context_text}
"""
    
    # 5. Generate Answer
    # Format history
    history_text = ""
    for role, msg in history:
        history_text += f"{role}: {msg}\n"
    history_text += f"User: {user_query}"

    prompt = prompts.RAG_ANSWER_PROMPT.format(
        context_text=full_context, 
        history_text=history_text, 
        user_query=user_query
    )

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
