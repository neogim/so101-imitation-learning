import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from compare_servo_snapshot_diagnostics import flatten


class CompareServoDiagnosticsTests(unittest.TestCase):
    def test_flattens_percentiles_and_identity(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"a.json"
            path.write_text(json.dumps({"camera_condition":"wrist", "summary":{
                "samples":100,"successful":100,"errors":0,"over_100ms":0,
                "total_duration_ms":{"mean":4,"p50":4,"p95":5,"p99":6,"max":7},
                "register_duration_ms":{}}, "camera":{"wrist":{"identity":{
                    "unique_id":"usb", "localized_name":"InnoMaker"}}}}))
            row = flatten("C", path)
            self.assertEqual(row["condition"], "C")
            self.assertEqual(row["snapshot_p99_ms"], 6)
            self.assertIn("InnoMaker", row["camera_identities_json"])


if __name__ == "__main__":
    unittest.main()
