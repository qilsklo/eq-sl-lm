
import llm
import standardscraper
import json
from unittest.mock import MagicMock

# Mock standardscraper.client.search to avoid needing actual DB
# This allows us to test the logic without relying on the state of the DB
standardscraper.client = MagicMock()
standardscraper.embedding_fn = MagicMock()
standardscraper.embedding_fn.encode_queries.return_value = [[0.1] * 768]

# Mock PDF Response
pdf_hit = {
    'entity': {
        'text': 'Seismic tomography is like a CT scan for the earth.',
        'title': 'Intro to Seismology',
        'page_num': 42,
        'author': 'Smith, J.',
        'publication_year': 2020
    },
    'distance': 0.9
}

# Mock Web Response
web_hit = {
    'entity': {
        'text': 'The USGS monitors earthquakes worldwide.',
        'url': 'https://usgs.gov',
        'site_name': 'USGS',
        'heading': 'Monitoring',
        'crawl_date': '2023-01-01'
    },
    'distance': 0.8
}

standardscraper.client.search.side_effect = [[[pdf_hit]], [[web_hit]]]

# Test search_knowledge_base
print("Testing search_knowledge_base...")
results = llm.search_knowledge_base("test")
print(json.dumps(results, indent=2))

# Verify Structure
assert len(results) == 2
assert results[0]['type'] == 'PDF'
assert results[0]['title'] == 'Intro to Seismology'
assert results[1]['type'] == 'WEB'
assert results[1]['site_name'] == 'USGS'

print("\nSUCCESS: search_knowledge_base returns correct structured data.")

# Test query_rag context formatting (Concept Mode)
# We need to mock get_search_params to return mode="concept"
llm.get_search_params = MagicMock(return_value={
    "mode": "concept",
    "semantic_query": "test",
    "start_date": None,
    "end_date": None,
    "user_coordinates": None,
    "min_magnitude": None
})

# Mock earthquake_data.manager
import earthquake_data
earthquake_data.manager = MagicMock()
earthquake_data.manager.get_context_for_llm.return_value = {"events": []}
earthquake_data.manager.fetch_feed.return_value = []

# Mock genai to avoid actual API call, we just want to check the prompt construction if possible
# But query_rag constructs prompt inside. We can inspect the call to model.generate_content
llm.genai = MagicMock()
mock_model = MagicMock()
llm.genai.GenerativeModel.return_value = mock_model
mock_model.generate_content.return_value.text = "Mock Answer"

# Reset side_effect for the next call in query_rag
standardscraper.client.search.side_effect = [[[pdf_hit]], [[web_hit]]]

print("\nTesting query_rag (Concept Mode)...")
llm.query_rag("What is tomography?", [], "fake_key")

# Inspect the prompt passed to generate_content
call_args = mock_model.generate_content.call_args
prompt_sent = call_args[0][0]

print("\n--- Prompt Sent (Snippet) ---")
print(prompt_sent[:500] + "...")
print("\n--- End Snippet ---")

# Check if JSON structure is in the prompt
if ' "type": "PDF"' in prompt_sent and ' "title": "Intro to Seismology"' in prompt_sent:
    print("\nSUCCESS: Prompt contains structured JSON context.")
else:
    print("\nFAILURE: Prompt does not contain structured JSON context.")

