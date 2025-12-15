import unittest
import json
import datetime
from earthquake_data import EarthquakeEvent, is_likely_aftershock, calculate_distance, format_time_relative

class TestEarthquakeLM(unittest.TestCase):
    def test_distance_calculation(self):
        # San Francisco to Los Angeles
        sf_lat, sf_lon = 37.7749, -122.4194
        la_lat, la_lon = 34.0522, -118.2437
        dist = calculate_distance(sf_lat, sf_lon, la_lat, la_lon)
        self.assertAlmostEqual(dist, 559, delta=5) # Approx 559 km

    def test_aftershock_logic(self):
        main_shock = EarthquakeEvent(
            event_id="main",
            origin_time_utc="2025-01-01T10:00:00+00:00",
            last_updated_utc="2025-01-01T10:00:00+00:00",
            magnitude=7.1,
            magnitude_type="mw",
            depth_km=10,
            latitude=35.0,
            longitude=-117.0,
            place_description="Ridgecrest",
            review_status="reviewed",
            tsunami_flag=False,
            version=1
        )
        
        aftershock = EarthquakeEvent(
            event_id="after",
            origin_time_utc="2025-01-01T11:00:00+00:00", # 1 hour later
            last_updated_utc="2025-01-01T11:00:00+00:00",
            magnitude=5.4,
            magnitude_type="mw",
            depth_km=10,
            latitude=35.1, # Nearby
            longitude=-117.1,
            place_description="Ridgecrest Area",
            review_status="automatic",
            tsunami_flag=False,
            version=1
        )
        
        unrelated = EarthquakeEvent(
            event_id="unrelated",
            origin_time_utc="2025-01-01T11:00:00+00:00",
            last_updated_utc="2025-01-01T11:00:00+00:00",
            magnitude=6.0,
            magnitude_type="mw",
            depth_km=10,
            latitude=40.0, # Far away
            longitude=-120.0,
            place_description="NorCal",
            review_status="automatic",
            tsunami_flag=False,
            version=1
        )

        events = [main_shock, aftershock, unrelated]
        
        self.assertTrue(is_likely_aftershock(aftershock, events))
        self.assertFalse(is_likely_aftershock(main_shock, events)) # Main shock is largest
        self.assertFalse(is_likely_aftershock(unrelated, events)) # Too far

    def test_relative_time(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        ten_mins_ago = (now - datetime.timedelta(minutes=10)).isoformat()
        self.assertEqual(format_time_relative(ten_mins_ago), "10 minutes ago")

if __name__ == '__main__':
    unittest.main()
