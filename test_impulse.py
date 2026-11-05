import tempfile
import unittest
from pathlib import Path
import numpy as np
from GameSimulator.elFaroGames.single import simulate, plot_simulation


class ImpulseTests(unittest.TestCase):
    def test_individual_limits_and_probabilities(self):
        limits = [0, 0.5, 2, 4]
        a = simulate(4, 1000, capacity=3, learning_rate=0,
                     initial_forecast=3, impulse_max=limits)
        np.testing.assert_array_equal(a.impulse_max, limits)
        self.assertTrue(np.all(a.impulses >= 0))
        self.assertTrue(np.all(a.impulses <= np.array(limits)))
        np.testing.assert_array_equal(a.impulses[:, 0], 0)
        np.testing.assert_allclose(a.probabilities, np.tile([0, 0, 0.5, 0.75], (1001, 1)))
        self.assertFalse(a.attendance[:, :2].any())
        self.assertTrue(0.45 < a.attendance[:, 2].mean() < 0.55)
        self.assertTrue(0.70 < a.attendance[:, 3].mean() < 0.80)

    def test_equal_and_zero_vectors_match_scalar(self):
        for limit in (0, 5):
            a = simulate(4, 30, impulse_max=limit)
            b = simulate(4, 30, impulse_max=[limit] * 4)
            np.testing.assert_array_equal(a.impulses, b.impulses)
            np.testing.assert_array_equal(a.forecasts, b.forecasts)
            np.testing.assert_array_equal(a.probabilities, b.probabilities)

    def test_vector_validation_and_zero_capacity(self):
        for limits in ([0, 2], [[0, 1, 2]], [0, -1, 2], [0, np.nan, 2], [0, np.inf, 2]):
            with self.assertRaises(ValueError):
                simulate(3, 2, impulse_max=limits)
        a = simulate(3, 10, capacity=0, impulse_max=[0, 2, 10])
        self.assertFalse(a.attendance.any())
        self.assertFalse(a.probabilities.any())

    def test_zero_impulse_matches_single(self):
        a = simulate(10, 50, seed=9)
        b = simulate(10, 50, seed=9, impulse_max=0)
        np.testing.assert_array_equal(a.forecasts, b.forecasts)
        np.testing.assert_array_equal(a.attendance, b.attendance)

    def test_draws_decisions_and_history(self):
        a = simulate(10, 50, seed=9, impulse_max=3)
        b = simulate(10, 50, seed=9, impulse_max=3)
        np.testing.assert_array_equal(a.impulses, b.impulses)
        np.testing.assert_array_equal(a.attendance, b.attendance)
        self.assertTrue(np.all((a.impulses >= 0) & (a.impulses < 3)))
        np.testing.assert_allclose(a.decision_forecasts, np.maximum(0, a.forecasts[:-1] - a.impulses))
        np.testing.assert_array_equal(a.attendance, a.decision_forecasts + 1 <= a.capacity)
        observed = a.attendance.sum(axis=1)[:, None] - a.attendance.astype(int)
        np.testing.assert_allclose(a.forecasts[1:], 0.9 * a.forecasts[:-1] + 0.1 * observed)
        payoff = np.where(a.attendance.sum(axis=1) <= a.capacity, 1, -1)
        np.testing.assert_array_equal(a.rewards, a.attendance * payoff[:, None])

    def test_probability_and_switching(self):
        a = simulate(3, 2000, capacity=2, learning_rate=0,
                     initial_forecast=[0, 1, 2], impulse_max=2)
        np.testing.assert_allclose(a.probabilities, np.tile([1, 1, 0.5], (2001, 1)))
        self.assertTrue(a.attendance[:, :2].all())
        self.assertTrue(0.45 < a.attendance[:, 2].mean() < 0.55)
        np.testing.assert_array_equal(a.forecasts, np.tile([0, 1, 2], (2001, 1)))

    def test_extremes_and_invalid_inputs(self):
        for capacity, expected in ((0, False), (3, True)):
            a = simulate(3, 10, capacity=capacity, impulse_max=100)
            self.assertTrue(np.all(a.attendance == expected))
            self.assertTrue(np.all(a.probabilities == float(expected)))
        for value in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                simulate(impulse_max=value)

    def test_reports_and_zero_rounds(self):
        with tempfile.TemporaryDirectory() as folder:
            for rounds in (0, 4):
                a = simulate(3, rounds, impulse_max=2)
                self.assertEqual(a.impulses.shape, (rounds, 3))
                path = Path(folder) / 'impulse.html'
                fig = plot_simulation(a, path)
                self.assertEqual(len(fig.data), 16)
                self.assertIn('random impulse', path.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
