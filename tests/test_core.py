import unittest
from analytics_engine import calculate_attendance, track_attendance, handle_detection_dropouts

class TestAnalyticsEngine(unittest.TestCase):
    def test_calculate_attendance(self):
        self.assertEqual(calculate_attendance([], 10), 0.0)
        self.assertEqual(calculate_attendance([1, 2, 3], 10), 0.3)
        self.assertEqual(calculate_attendance([1, 2, 3], None), "Not set")

    def test_track_attendance(self):
        self.assertEqual(track_attendance(10, 30), 10/30)
        self.assertEqual(track_attendance(20, 60), 20/60)

    def test_handle_detection_dropouts(self):
        profiles = [1, 2, 3]
        self.assertEqual(handle_detection_dropouts(profiles, 10), profiles)

if __name__ == "__main__":
    unittest.main()
