import unittest
import numpy as np
from GameSimulator.elFaroGames.mixed import simulate, plot_simulation
import tempfile
from pathlib import Path


class MixedStrategyTests(unittest.TestCase):
    def test_reproducible_and_rewards(self):
        a = simulate(17, 80, seed=9)
        b = simulate(17, 80, seed=9)
        np.testing.assert_array_equal(a.attendance, b.attendance)
        np.testing.assert_array_equal(a.probabilities, b.probabilities)
        self.assertEqual(a.probabilities.shape, (81, 17))
        self.assertTrue(np.all((a.probabilities >= 0) & (a.probabilities <= 1)))
        expected = np.where(a.attendance.sum(axis=1) <= a.capacity, 1, -1)
        np.testing.assert_array_equal(a.rewards, a.attendance * expected[:, None])

    def test_learning_direction(self):
        full = simulate(10, 20, capacity=10)
        empty = simulate(10, 20, capacity=0)
        self.assertTrue(np.all(np.diff(full.probabilities, axis=0) > 0))
        self.assertTrue(np.all(np.diff(empty.probabilities, axis=0) < 0))

    def test_capacity_boundary_counts_own_seat(self):
        result = simulate(2, 1, capacity=1, initial_probability=[0.999, 0.001], seed=42, learning_rate=1)
        np.testing.assert_array_equal(result.attendance[0], [True, False])
        np.testing.assert_array_equal(result.rewards[0], [1, 0])
        self.assertGreater(result.probabilities[1, 0], 0.5)
        self.assertLess(result.probabilities[1, 1], 0.5)

    def test_no_learning_and_zero_rounds(self):
        result = simulate(2, 10, initial_probability=[0.2, 0.8], learning_rate=0)
        np.testing.assert_allclose(result.probabilities, np.tile([0.2, 0.8], (11, 1)))
        self.assertEqual(simulate(2, 0).probabilities.shape, (1, 2))

    def test_invalid_parameters(self):
        for kwargs in ({'num_players': 0}, {'num_rounds': -1}, {'capacity': 101},
                       {'temperature': 0}, {'learning_rate': 2}, {'initial_probability': 1},
                       {'initial_probability': [0.5]}, {'temperature': float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                simulate(**kwargs)

    def test_report(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'report.html'
            fig = plot_simulation(simulate(3, 4), path)
            self.assertEqual(len(fig.data), 10)
            self.assertEqual(len(fig.data[0].y), 5)
            self.assertIn('Plotly.newPlot', path.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
