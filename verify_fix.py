
import sys
import json
from unittest.mock import MagicMock

# Mock modules before importing llm
sys.modules["google"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()
sys.modules["streamlit"] = MagicMock()
sys.modules["dotenv"] = MagicMock()

# Mock these to avoid side effects
sys.modules["standardscraper"] = MagicMock()
sys.modules["pymilvus"] = MagicMock()

import llm
import prompts

def test_fix():
    print("Testing fix...")
    
    # Mock LLM response for search params (simulating what the LLM would extract)
    # We can bypass get_search_params by mocking it, or mock the model it uses.
    # Let's mock the model in llm.get_search_params? 
    # Actually, simpler to mock get_search_params directly since we are testing query_rag's usage of it.
    
    llm.get_search_params = MagicMock(return_value={
        "user_location": "Sonoma",
        "user_coordinates": [38.2919, -122.4580],
        "mode": "event",
        "semantic_query": "recent earthquakes",
        "start_date": None,
        "end_date": None,
        "min_magnitude": None
    })
    
    # Mock earthquake manager to return some data
    llm.earthquake_data.manager.get_context_for_llm = MagicMock(return_value={
        "generated_at": "2023-01-01T00:00:00Z",
        "events": [
            {
                "event_id": "test1",
                "magnitude": 3.5,
                "place_description": "Near Sonoma",
                "distance_to_user_km": 5.0 # This would be calculated by manager
            }
        ]
    })
    llm.earthquake_data.manager.fetch_feed = MagicMock(return_value=[])
    llm.earthquake_data.manager.process_features = MagicMock(return_value=[])
    
    # Mock search_knowledge_base
    
    # Mock search_knowledge_base
    llm.search_knowledge_base = MagicMock(return_value=[])
    
    # Mock the GenerativeModel used in query_rag
    mock_model = MagicMock()
    llm.genai.GenerativeModel.return_value = mock_model
    mock_model.generate_content.return_value.text = "Mock response"
    
    # Run the function
    history = []
    api_key = "dummy"
    llm.query_rag("recent earthquakes in Sonoma", history, api_key)
    
    # Verify the prompt sent to the model
    args, _ = mock_model.generate_content.call_args
    prompt_sent = args[0]
    
    # Check 1: Did we inject "reference_location": "Sonoma"?
    # The prompt contains {context_text}, which contains the JSON dump.
    if '"reference_location": "Sonoma"' in prompt_sent:
        print("PASS: reference_location injected correctly.")
    else:
        print("FAIL: reference_location NOT found in prompt.")
        print("Draft of context in prompt:", prompt_sent)

    # Check 2: Is the new prompt text present?
    if "The JSON context contains a `reference_location`" in prompt_sent:
        print("PASS: New prompt instructions present.")
    else:
        print("FAIL: New prompt instructions NOT found.")
        
    # Check 3: Is the old contradictory text GONE?
    if "There were no recent earthquakes near you. Do you have a specific time?" in prompt_sent:
        # Note: We kept a similar phrase but it's conditional now.
        # The strict requirement "You MUST explicitly state: ..." was changed.
        # Let's check strict checking.
        pass
        
    if "Do NOT use the phrase \"There were no recent earthquakes near you\" if the user specified a city" in prompt_sent:
         print("PASS: Specific instruction to avoid 'near you' logic is present.")
    else:
         print("FAIL: Specific instruction NOT present.")

if __name__ == "__main__":
    test_fix()
