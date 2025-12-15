import requests
import json
import datetime
import math
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any
from enum import Enum

# --- Constants ---
USGS_API_BASE = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"
CACHE_FILE = "earthquake_cache.json"

class ReviewStatus(Enum):
    AUTOMATIC = "automatic"
    REVIEWED = "reviewed"

@dataclass
class EarthquakeEvent:
    event_id: str
    origin_time_utc: str  # ISO 8601
    last_updated_utc: str # ISO 8601
    magnitude: Optional[float]
    magnitude_type: Optional[str]
    depth_km: float
    latitude: float
    longitude: float
    place_description: str
    review_status: str
    tsunami_flag: bool
    source: str = "USGS"
    version: int = 1

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

# --- Deterministic Logic ---

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Haversine formula to calculate distance in km.
    """
    R = 6371.0 # Radius of Earth in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def is_likely_aftershock(event: EarthquakeEvent, prior_events: List[EarthquakeEvent]) -> bool:
    """
    Heuristics:
    - Within 7 days
    - Within 50 km
    - Lower magnitude
    """
    event_time = datetime.datetime.fromisoformat(event.origin_time_utc)
    
    for prior in prior_events:
        if prior.event_id == event.event_id:
            continue
            
        prior_time = datetime.datetime.fromisoformat(prior.origin_time_utc)
        
        # Check time window (within 7 days before event)
        delta = event_time - prior_time
        if not (datetime.timedelta(days=0) < delta <= datetime.timedelta(days=7)):
            continue
            
        # Check magnitude (must be smaller)
        if event.magnitude is None or prior.magnitude is None or event.magnitude >= prior.magnitude:
            continue
            
        # Check distance
        dist = calculate_distance(event.latitude, event.longitude, prior.latitude, prior.longitude)
        if dist <= 50:
            return True
            
    return False

def format_time_relative(timestamp_str: str) -> str:
    """Returns a human readable relative time string."""
    dt = datetime.datetime.fromisoformat(timestamp_str)
    now = datetime.datetime.now(datetime.timezone.utc)
    diff = now - dt
    
    if diff < datetime.timedelta(minutes=1):
        return "just now"
    elif diff < datetime.timedelta(hours=1):
        return f"{int(diff.total_seconds() / 60)} minutes ago"
    elif diff < datetime.timedelta(hours=24):
        return f"{int(diff.total_seconds() / 3600)} hours ago"
    else:
        return f"{diff.days} days ago"

# --- Data Ingestion ---

class EarthquakeManager:
    def __init__(self):
        self.cache: Dict[str, EarthquakeEvent] = {}
        self.last_fetch_time: Optional[datetime.datetime] = None
        self.load_cache()

    def load_cache(self):
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r') as f:
                    data = json.load(f)
                    # We could save/load last_fetch_time too, but for now it resets on restart
                    # which is safer (forces a fetch).
                    events_data = data.get("events", {})
                    for k, v in events_data.items():
                        self.cache[k] = EarthquakeEvent.from_dict(v)
            except Exception as e:
                print(f"Error loading cache: {e}")

    def save_cache(self):
        try:
            with open(CACHE_FILE, 'w') as f:
                # Save wrapper object
                json.dump({
                    "events": {k: v.to_dict() for k, v in self.cache.items()},
                    "last_saved": datetime.datetime.now(datetime.timezone.utc).isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def fetch_feed(self, feed="all_day"):
        url = f"{USGS_API_BASE}/{feed}.geojson"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            self.last_fetch_time = datetime.datetime.now(datetime.timezone.utc)
            return data.get("features", [])
        except Exception as e:
            print(f"Error fetching feed {feed}: {e}")
            raise e # Re-raise to let caller know

    def process_features(self, features):
        new_events = []
        for feat in features:
            props = feat["properties"]
            geom = feat["geometry"]
            
            event_id = props["code"] # Using 'code' as unique ID per spec
            updated_ms = props["updated"]
            
            # Convert ms timestamps to ISO 8601
            origin_time = datetime.datetime.fromtimestamp(props["time"] / 1000, datetime.timezone.utc).isoformat()
            last_updated = datetime.datetime.fromtimestamp(updated_ms / 1000, datetime.timezone.utc).isoformat()
            
            # Versioning check
            existing = self.cache.get(event_id)
            version = 1
            if existing:
                existing_updated_dt = datetime.datetime.fromisoformat(existing.last_updated_utc)
                new_updated_dt = datetime.datetime.fromisoformat(last_updated)
                
                if new_updated_dt <= existing_updated_dt:
                    continue # No update needed
                version = existing.version + 1

            event = EarthquakeEvent(
                event_id=event_id,
                origin_time_utc=origin_time,
                last_updated_utc=last_updated,
                magnitude=props.get("mag"),
                magnitude_type=props.get("magType"),
                depth_km=geom["coordinates"][2],
                latitude=geom["coordinates"][1],
                longitude=geom["coordinates"][0],
                place_description=props["place"],
                review_status=props["status"],
                tsunami_flag=bool(props["tsunami"]),
                version=version
            )
            
            self.cache[event_id] = event
            new_events.append(event)
            
        if new_events:
            self.save_cache()
            
        return new_events

    def get_latest_events(self, limit=5) -> List[EarthquakeEvent]:
        # Sort by origin time descending
        sorted_events = sorted(self.cache.values(), key=lambda x: x.origin_time_utc, reverse=True)
        return sorted_events[:limit]

    def get_event(self, event_id) -> Optional[EarthquakeEvent]:
        return self.cache.get(event_id)

    def get_context_for_llm(self, event_id=None, user_lat=None, user_lon=None, min_magnitude=None, date_filter=None, limit=3):
        """
        Constructs the structured JSON context for the LLM.
        If event_id is provided, focuses on that event.
        Otherwise, provides a summary of recent activity.
        """
        context = {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "feed_status": "ok",
            "events": []
        }
        
        if self.last_fetch_time:
             time_since = datetime.datetime.now(datetime.timezone.utc) - self.last_fetch_time
             if time_since > datetime.timedelta(minutes=10):
                 context["feed_status"] = "stale"
                 context["feed_last_updated"] = self.last_fetch_time.isoformat()
        else:
             context["feed_status"] = "unknown"
        
        events_to_include = []
        if event_id and event_id in self.cache:
            events_to_include.append(self.cache[event_id])
        else:
            # Filter by magnitude if requested
            candidates = self.cache.values()
            if min_magnitude is not None:
                candidates = [e for e in candidates if e.magnitude is not None and e.magnitude >= min_magnitude]
            
            # Sort by time
            sorted_candidates = sorted(candidates, key=lambda x: x.origin_time_utc, reverse=True)
            events_to_include = sorted_candidates[:limit]
            
        # Get all events for aftershock check (naive approach, using all cache)
        # In a real DB we'd query efficiently. Here we just pass the list.
        all_events = list(self.cache.values())

        for evt in events_to_include:
            evt_data = evt.to_dict()
            
            # Derived facts
            evt_data["is_likely_aftershock"] = is_likely_aftershock(evt, all_events)
            evt_data["relative_time"] = format_time_relative(evt.origin_time_utc)
            
            if user_lat is not None and user_lon is not None:
                dist = calculate_distance(user_lat, user_lon, evt.latitude, evt.longitude)
                evt_data["distance_to_user_km"] = round(dist, 1)
            
            context["events"].append(evt_data)
            
        return context

# Singleton instance
manager = EarthquakeManager()
