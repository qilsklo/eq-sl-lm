
import earthquake_data
import datetime

print("--- Loading Month Data ---")
try:
    print("Fetching 'all_month' feed...")
    features = earthquake_data.manager.fetch_feed("all_month")
    print(f"Fetched {len(features)} events.")
    
    print("Processing features...")
    new_events = earthquake_data.manager.process_features(features)
    print(f"Processed. Cache now contains {len(earthquake_data.manager.cache)} events.")
    
except Exception as e:
    print(f"Error loading month data: {e}")
