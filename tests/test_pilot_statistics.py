import unittest

from scripts.analyze_pilot_results import (
    exact_mcnemar,
    weighted_task_accuracy,
    wilson_interval,
)


class PilotStatisticsTests(unittest.TestCase):
    def test_wilson_interval_handles_empty_and_boundary_samples(self):
        self.assertIsNone(wilson_interval(0, 0))
        interval = wilson_interval(0, 10)
        self.assertAlmostEqual(interval["lower"], 0.0)
        self.assertAlmostEqual(interval["upper"], 0.2775, places=3)
        all_correct = wilson_interval(10, 10)
        self.assertAlmostEqual(all_correct["lower"], 0.7225, places=3)
        self.assertAlmostEqual(all_correct["upper"], 1.0)

    def test_wilson_interval_rejects_invalid_counts(self):
        with self.assertRaises(ValueError):
            wilson_interval(3, 2)
        with self.assertRaises(ValueError):
            wilson_interval(-1, 2)

    def test_exact_mcnemar_is_two_sided_and_handles_no_discordance(self):
        self.assertEqual(exact_mcnemar(0, 3)["two_sided_exact_p_value"], 0.25)
        self.assertEqual(exact_mcnemar(3, 0)["two_sided_exact_p_value"], 0.25)
        self.assertEqual(exact_mcnemar(0, 0)["two_sided_exact_p_value"], 1.0)
        self.assertAlmostEqual(exact_mcnemar(4, 43)["two_sided_exact_p_value"], 0.0, places=7)

    def test_weighted_estimate_uses_explicit_task_mix(self):
        task_results = {
            task: {"accuracy": accuracy}
            for task, accuracy in {
                "math": 0.1,
                "mcq": 0.7,
                "extract": 0.75,
                "reformat": 0.2,
                "classify": 0.4375,
            }.items()
        }
        result = weighted_task_accuracy(task_results)
        self.assertEqual(result["population_total"], 300)
        self.assertAlmostEqual(result["estimated_correct"], 131.85)
        self.assertAlmostEqual(result["estimated_accuracy"], 131.85 / 300)
        self.assertIn("not measured full-test accuracy", result["label"])


if __name__ == "__main__":
    unittest.main()
