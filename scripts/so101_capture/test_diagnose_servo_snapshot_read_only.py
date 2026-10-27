import unittest

from diagnose_servo_snapshot_read_only import measure_once, summarize


class FakeBus:
    def __init__(self, fail=None):
        self.fail = fail

    def sync_read(self, register, normalize=True, num_retry=0):
        if register == self.fail:
            raise TimeoutError("synthetic timeout")
        return dict.fromkeys(range(6), 1)


class ServoSnapshotDiagnosticTests(unittest.TestCase):
    def test_records_order_and_threshold_without_writes(self):
        ticks = iter([0, .010, .020, .020, .040, .040, .060, .060, .110])
        row = measure_once(FakeBus(), clock=lambda: next(ticks))
        self.assertEqual(row["result"], "ok")
        self.assertTrue(row["threshold_exceeded"])
        self.assertAlmostEqual(row["total_duration_ms"], 110)
        self.assertEqual([x["register"] for x in row["register_reads"]],
                         ["Present_Position", "Goal_Position", "Torque_Enable", "Status"])
        self.assertEqual(summarize([row])["over_100ms"], 1)

    def test_failed_register_is_distinguished(self):
        ticks = iter([0, .01, .02, .02, .03])
        row = measure_once(FakeBus("Goal_Position"), clock=lambda: next(ticks))
        self.assertEqual(row["result"], "error")
        self.assertEqual(row["failed_register"], "Goal_Position")
        self.assertEqual(summarize([row])["errors"], 1)


if __name__ == "__main__":
    unittest.main()
