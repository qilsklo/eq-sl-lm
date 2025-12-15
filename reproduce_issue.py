
import llm
import prompts
import earthquake_data
import json

queries = [
    "Why does Cobb get so many eq's?",
    "what just happened"
]

print("--- Testing Prompt Construction ---")

for query in queries:
    print(f"\nQuery: {query}")
    
    # Mock history
    history = []

    # Get context
    context_data = earthquake_data.manager.get_context_for_llm(limit=5)
    event_context_json = json.dumps(context_data, indent=2)

    # Get docs
    docs = llm.perform_vector_search(query, limit=3)
    safety_context_text = "\n\n".join([f"[DOC {i+1}] {doc}" for i, doc in enumerate(docs)])

    full_context = f"""
--- EVENT DATA (Authoritative) ---
{event_context_json}

--- SAFETY DOCS (Reference) ---
{safety_context_text}
"""

    history_text = f"User: {query}"

    prompt = prompts.RAG_ANSWER_PROMPT.format(
        context_text=full_context, 
        history_text=history_text, 
        user_query=query
    )
    
    # Print the "System Contract" part of the prompt to verify changes
    start_idx = prompt.find("--- SYSTEM CONTRACT ---")
    end_idx = prompt.find("--- RESPONSE TEMPLATE ---")
    print(prompt[start_idx:end_idx])
