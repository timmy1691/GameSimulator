import unittest
from game import (Game, FixedStrategy, RandomStrategy, WinStayLoseSwitch,
                  split_prize, winner_reward, Outcome)


class Tests(unittest.TestCase):
    def test_rewards(self):
        for reward, expected in [(winner_reward(3, -1), [3, -1, -1]),
                                 (split_prize(10), [10, 0, 0])]:
            g = Game({'x': FixedStrategy('A'), 'y': FixedStrategy('B'),
                      'z': FixedStrategy('B')}, reward=reward)
            self.assertEqual(list(g.step().rewards.values()), expected)

    def test_tie_and_empty(self):
        g = Game({'x': FixedStrategy('A'), 'y': FixedStrategy('B')})
        self.assertFalse(g.step().outcome.winners)
        g = Game({'x': FixedStrategy('A')}, reward=split_prize())
        r = g.step()
        self.assertEqual(r.outcome.winners, {'B'})
        self.assertEqual(r.rewards['x'], 0)

    def test_feedback_boundary(self):
        seen = []
        class Spy:
            def choose(self, observation, actions, rng):
                seen.append(observation)
                return 'A'
        g = Game({'spy': Spy(), 'y': FixedStrategy('B'), 'z': FixedStrategy('B')})
        g.run(2)
        self.assertEqual(dict(seen[0].public), {})
        self.assertEqual(dict(seen[1].public), {'winner': 'A', 'minority_count': 1})
        self.assertEqual(seen[1].own_reward, 1)
        self.assertFalse(hasattr(seen[1], 'players'))

    def test_reproducible(self):
        def simulate():
            return Game({str(i): RandomStrategy() for i in range(11)}, seed=8).run(50)
        self.assertEqual(simulate(), simulate())

    def test_custom_game(self):
        def majority(choices, actions):
            counts = {a: sum(v == a for v in choices.values()) for a in actions}
            return Outcome(counts, frozenset(a for a in actions
                                            if counts[a] == max(counts.values())))
        g = Game({'x': FixedStrategy('A'), 'y': FixedStrategy('B'),
                  'z': FixedStrategy('B')}, rule=majority,
                 feedback=lambda r: {'winning_actions': tuple(r.outcome.winners)})
        self.assertEqual(g.step().rewards['y'], 1)

    def test_invalid_action(self):
        with self.assertRaises(ValueError):
            Game({'x': FixedStrategy('C')}).step()


if __name__ == '__main__':
    unittest.main()
