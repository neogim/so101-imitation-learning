import importlib.util
import itertools
import unittest
from pathlib import Path

import numpy as np

PATH = Path(__file__).with_name("select_ik_trajectory_dp_offline.py")
SPEC = importlib.util.spec_from_file_location("trajectory_dp", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def candidate(value, seed="a", residual=0.1):
    return {"motor": [float(value), 0.0, 0.0, 0.0, 0.0], "seed": seed, "position_residual_mm": residual,
            "orientation_change_deg": 0.0, "active_bound_joints": []}


class TrajectoryDPTests(unittest.TestCase):
    def test_deduplicate_keeps_lowest_residual_deterministically(self):
        rows = [
            {**candidate(1.0, "z", .2), "candidate_motor_deg": [1, 0, 0, 0, 0]},
            {**candidate(1.0, "a", .1), "candidate_motor_deg": [1, 0, 0, 0, 0]},
        ]
        result = MODULE.deduplicate(rows)
        self.assertEqual(len(result), 1); self.assertEqual(result[0]["seed"], "a")

    def test_missing_layers_form_separate_segments(self):
        self.assertEqual(MODULE.contiguous_segments([[1], [2], [], [3], []]), [(0, 2), (3, 4)])

    def test_first_order_dp_beats_greedy_toy(self):
        layers = [[candidate(0, "a"), candidate(3, "b")],
                  [candidate(1, "a"), candidate(3, "b")],
                  [candidate(5, "a")]]
        actions = np.zeros((3, 5)); times = np.array([0, .05, .1])
        _, greedy = MODULE.select_greedy(layers, actions, times)
        _, dp = MODULE.select_first_order(layers, actions, times)
        self.assertLessEqual(dp, greedy)

    def test_second_order_matches_bruteforce_toy(self):
        layers = [[candidate(0), candidate(1)], [candidate(1), candidate(3)], [candidate(2), candidate(6)]]
        actions = np.zeros((3, 5)); times = np.array([0, .05, .1])
        path, cost = MODULE.select_second_order(layers, actions, times)
        brute = []
        for choices in itertools.product(range(2), repeat=3):
            selected = [layers[i][choices[i]] for i in range(3)]
            value = sum(MODULE.node_cost(selected[i], actions[i]) for i in range(3))
            value += MODULE.velocity_cost(selected[0], selected[1], .05)
            value += MODULE.velocity_cost(selected[1], selected[2], .05)
            value += MODULE.acceleration_cost(*selected, .05, .05)
            brute.append((value, choices))
        self.assertAlmostEqual(cost, min(brute)[0]); self.assertEqual(tuple(path), min(brute)[1])

    def test_acceleration_uses_actual_timestamps(self):
        a, b, c = candidate(0), candidate(1), candidate(3)
        uniform = MODULE.acceleration_cost(a, b, c, .05, .05)
        nonuniform = MODULE.acceleration_cost(a, b, c, .05, .1)
        self.assertNotEqual(uniform, nonuniform)

    def test_quality_gate_rejects_large_jump(self):
        selected = [candidate(0), candidate(6), candidate(6)]
        actions = np.zeros((3, 5), dtype=float); times = np.array([0, .05, .1]); frames = np.array([1, 2, 3])
        result = MODULE.path_metrics(selected, actions, times, frames, np.full(5, -180), np.full(5, 180))
        self.assertFalse(result["quality_pass"]); self.assertEqual(result["jump_peak"]["value"], 6)


if __name__ == "__main__":
    unittest.main()
