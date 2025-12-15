import time
import datetime
import earthquake_data

def run_updater(interval_seconds=300):
    """
    Continuously fetches the 'all_hour' feed to keep the cache fresh.
    """
    print(f"Starting Earthquake Updater Service (Interval: {interval_seconds}s)")
    
    while True:
        try:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] Fetching 'all_hour' feed...")
            
            # Fetch and process data
            # This automatically updates the cache file via the manager
            features = earthquake_data.manager.fetch_feed("all_hour")
            earthquake_data.manager.process_features(features)
            
            print(f"[{now}] Update complete. Cache contains {len(earthquake_data.manager.cache)} events.")
            
        except Exception as e:
            print(f"Error during update: {e}")
        
        time.sleep(interval_seconds)

if __name__ == "__main__":
    # Run every 5 minutes by default
    run_updater()
