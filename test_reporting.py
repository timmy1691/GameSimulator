import tempfile
import unittest
from pathlib import Path
import numpy as np
from GameSimulator.elFaroGames.single import simulate, plot_simulation


class PlayerReportTests(unittest.TestCase):
    def test_selector_and_cumulative_rewards(self):
        result = simulate(3, 4, capacity=2, initial_forecast=[0, 1, 2], impulse_max=[0, 1, 2])
        with tempfile.TemporaryDirectory() as folder:
            fig = plot_simulation(result, Path(folder) / 'report.html', player=2)
        reward_traces = [t for t in fig.data if t.name == 'Cumulative reward']
        self.assertEqual(len(reward_traces), 3)
        for p, trace in enumerate(reward_traces):
            np.testing.assert_array_equal(trace.y, np.r_[0, np.cumsum(result.rewards[:, p])])
            self.assertEqual(trace.y[-1], result.rewards[:, p].sum())
            self.assertEqual(list(trace.x), [0, 1, 2, 3, 4])
        self.assertEqual(fig.layout.updatemenus[0].active, 1)
        for p, button in enumerate(fig.layout.updatemenus[0].buttons):
            self.assertIn(f'max impulse: {p}', button.label)
            visible = list(button.args[0]['visible'])
            self.assertEqual(visible, [i // 5 == p for i in range(15)] + [True])
            self.assertIn(f'Player {p + 1}', button.args[1]['title']['text'])
        self.assertEqual([t.visible is not False for t in fig.data],
                         list(fig.layout.updatemenus[0].buttons[1].args[0]['visible']))

    def test_invalid_player(self):
        result = simulate(3, 0)
        for player in (0, 4, 1.5, True):
            with self.assertRaises(ValueError):
                plot_simulation(result, player=player)


if __name__ == '__main__':
    unittest.main()
