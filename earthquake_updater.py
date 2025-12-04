
import standardscraper

def update_hourly():
    print("Running hourly update...")
    chunks = standardscraper.fetch_earthquake_feed(standardscraper.HOUR_FEED)
    if not chunks:
        print("No data found in hourly feed.")
        return

    # Deduplicate
    unique_chunks = []
    for chunk in chunks:
        # Check if origin already exists
        res = standardscraper.client.query(
            collection_name=standardscraper.collection_name,
            filter=f'origin == "{chunk["origin"]}"',
            output_fields=["id"]
        )
        if not res:
            unique_chunks.append(chunk)
        else:
            print(f"Duplicate found, skipping: {chunk['origin']}")
            
    if unique_chunks:
        print(f"Storing {len(unique_chunks)} new earthquake records.")
        standardscraper.db_store(unique_chunks)
    else:
        print("No new unique earthquake records to store.")

if __name__ == "__main__":
    update_hourly()
