import tempfile
import unittest
from pathlib import Path
import numpy as np
from GameSimulator.elFaroGames.single import simulate, plot_simulation


class SingleStrategyTests(unittest.TestCase):
    def test_history_update_and_simultaneous_choices(self):
        result = simulate(3, 2, capacity=2, learning_rate=0.5,
                          initial_forecast=[0, 1, 2])
        np.testing.assert_array_equal(result.attendance, [[1, 1, 0], [1, 1, 0]])
        np.testing.assert_allclose(result.forecasts, [[0, 1, 2], [0.5, 1, 2], [0.75, 1, 2]])
        np.testing.assert_array_equal(result.rewards, [[1, 1, 0], [1, 1, 0]])

    def test_overcrowding_changes_next_decision(self):
        result = simulate(3, 3, capacity=2, learning_rate=1, initial_forecast=0)
        np.testing.assert_array_equal(result.attendance, [[1, 1, 1], [0, 0, 0], [1, 1, 1]])
        np.testing.assert_array_equal(result.rewards[0], [-1, -1, -1])

    def test_capacity_extremes(self):
        for players in (1, 10):
            self.assertFalse(simulate(players, 5, capacity=0).attendance.any())
            self.assertTrue(simulate(players, 5, capacity=players).attendance.all())

    def test_seed_zero_rounds_and_frozen_beliefs(self):
        a = simulate(10, 20, seed=4)
        b = simulate(10, 20, seed=4)
        np.testing.assert_array_equal(a.forecasts, b.forecasts)
        self.assertEqual(simulate(10, 0).forecasts.shape, (1, 10))
        frozen = simulate(3, 4, learning_rate=0, initial_forecast=[0, 1, 2])
        np.testing.assert_array_equal(frozen.forecasts, np.tile([0, 1, 2], (5, 1)))

    def test_invalid_parameters(self):
        for kwargs in ({'num_players': 0}, {'num_rounds': -1}, {'capacity': 101},
                       {'learning_rate': -1}, {'initial_forecast': -1},
                       {'initial_forecast': 100}, {'initial_forecast': [1, 2]},
                       {'initial_forecast': float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                simulate(**kwargs)

    def test_report(self):
        with tempfile.TemporaryDirectory() as folder:
            for rounds in (0, 4):
                path = Path(folder) / 'single.html'
                fig = plot_simulation(simulate(3, rounds), path)
                self.assertEqual(len(fig.data), 13)
                self.assertEqual(len(fig.data[0].y), rounds + 1)
                self.assertIn('Plotly.newPlot', path.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
