import sys
import os

# Add the 'src' folder to Python's search path so it can find pipeline_engine
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import unittest
import pandas as pd
import numpy as np

# Now this import will work correctly!
from pipeline_engine import compute_rolling_fatigue, validate_skating_content

# (Keep the rest of your test class code below...)

class TestPipelineEdgeCases(unittest.TestCase):
    
    def test_compute_rolling_fatigue_empty(self):
        """Test that rolling fatigue handles an empty list gracefully."""
        result_df = compute_rolling_fatigue([], window_size=30, fps=30.0)
        self.assertTrue(result_df.empty, "DataFrame should be empty for empty input.")
        self.assertIn("rolling_loss", result_df.columns, "Columns should still be structured correctly.")

    def test_compute_rolling_fatigue_valid(self):
        """Test that rolling fatigue computes correctly with sample data."""
        mock_data = [(i, 0.02 + (i * 0.001)) for i in range(50)]
        result_df = compute_rolling_fatigue(mock_data, window_size=10, fps=30.0)
        
        self.assertEqual(len(result_df), 50, "Length of output should match input size.")
        self.assertIn("timestamp_sec", result_df.columns, "Timestamp column should be present.")
        self.assertAlmostEqual(result_df["timestamp_sec"].iloc[30], 30.0 / 30.0, places=2)

    def test_validate_skating_content_too_short(self):
        """Test that short or empty video features are rejected."""
        short_df = pd.DataFrame({"frame": [1, 2, 3], "right_knee_angle": [45.0, 46.0, 47.0]})
        is_valid, msg = validate_skating_content(short_df)
        self.assertFalse(is_valid, "Short videos should fail validation.")
        self.assertIn("too short", msg.lower())

    def test_validate_skating_content_flat_line(self):
        """Test that static/non-skating videos with no movement variation are rejected."""
        # Create a dataframe with 60 frames but flat knee angles (no motion)
        flat_df = pd.DataFrame({
            "frame": range(60),
            "right_knee_angle": [45.0] * 60,
            "left_knee_angle": [45.0] * 60
        })
        is_valid, msg = validate_skating_content(flat_df)
        self.assertFalse(is_valid, "Videos without movement variation should fail validation.")
        self.assertIn("invalid content", msg.lower())

if __name__ == "__main__":
    unittest.main()