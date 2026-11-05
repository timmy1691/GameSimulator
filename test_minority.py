import unittest
from GameSimulator.krillion.minority import Config, simulate, write_report


class MinorityTests(unittest.TestCase):
    def test_rewards_history_and_available_knowledge(self):
        c = Config(players=15, rounds=40, blank_probability=.2, learning_probability=1)
        result = simulate(c)
        knowledge = [set(k) for k in result['initial_knowledge']]
        for r in result['rounds']:
            for p, a in enumerate(r['choices']):
                self.assertTrue(knowledge[p].intersection(r['offered']))
                if a is not None:
                    self.assertIn(a, knowledge[p])
                    self.assertIn(a, r['offered'])
                    self.assertEqual(r['rewards'][p], r['population'] / r['counts'][a])
                else:
                    self.assertEqual(r['rewards'][p], 0)
                learned = r['learned'][p]
                if learned is not None:
                    self.assertTrue(r['present'][p])
                    self.assertIn(learned, r['best'])
                    knowledge[p].add(learned)
                self.assertEqual(len(knowledge[p]), r['knowledge_sizes'][p])
            self.assertAlmostEqual(sum(r['rewards']), r['population'] * sum(n > 0 for n in r['counts'].values()))
        for p, history in enumerate(result['histories']):
            self.assertEqual([h['round'] for h in history], [r['round'] for r in result['rounds'] if r['present'][p]])

    def test_absent_and_blank_rounds(self):
        absent = simulate(Config(rounds=3, absence_probability=1))
        self.assertTrue(all(not h for h in absent['histories']))
        self.assertEqual(absent['initial_knowledge'], absent['final_knowledge'])
        blank = simulate(Config(rounds=3, absence_probability=0, blank_probability=1))
        self.assertTrue(all(len(h) == 3 for h in blank['histories']))
        self.assertTrue(all(r['population'] == 50 and not r['best'] for r in blank['rounds']))
        self.assertEqual(sum(blank['totals']), 0)

    def test_seed_and_single_answer(self):
        c = Config(players=1, rounds=3, answer_pool=1, answers_per_round=1,
                   initial_knowledge=1, absence_probability=0)
        self.assertEqual(simulate(c), simulate(c))
        self.assertEqual(simulate(c)['totals'], [3])

    def test_validation(self):
        for kwargs in ({'players': 0}, {'learning_probability': float('nan')},
                       {'answers_per_round': Config().answer_pool + 1}, {'initial_knowledge': 0}):
            with self.assertRaises(ValueError):
                simulate(Config(**kwargs))

    def test_player_report_scores_and_knowledge(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        result = simulate(Config(players=3, rounds=5))
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'report.html'
            fig = write_report(result, path)
            page = path.read_text(encoding='utf-8')
        self.assertEqual(len(fig.data), 5)
        score, knowledge, probabilities, table, attendance = fig.data
        self.assertEqual(list(score.x), list(range(6)))
        self.assertEqual(list(score.y), [0] + [r['cumulative'][0] for r in result['rounds']])
        self.assertEqual(knowledge.y[0], len(result['initial_knowledge'][0]))
        self.assertEqual(knowledge.y[-1], len(result['final_knowledge'][0]))
        self.assertEqual(probabilities.z[0][0], result['rounds'][0]['answer_probabilities'][0][0])
        self.assertEqual(len(fig.layout.updatemenus[0].buttons), 3)
        self.assertEqual(len(fig.layout.updatemenus[1].buttons), 5)
        self.assertIn('plotly_buttonclicked', page)
        self.assertIn('P(answer | participates)', page)
        self.assertEqual(list(attendance.y), [r['population'] for r in result['rounds']])

    def test_probability_snapshots(self):
        c = Config(players=8, rounds=20, blank_probability=.25, exploration=.2)
        result = simulate(c)
        for r in result['rounds']:
            for p in range(c.players):
                probabilities = r['answer_probabilities'][p]
                estimates = r['answer_estimates'][p]
                known = r['knowledge_before'][p]
                self.assertAlmostEqual(sum(probabilities), .75)
                available = [a for a in r['offered'] if a in known]
                maximum = max(e for e in estimates if e is not None)
                ties = estimates.count(maximum)
                for a, e, probability in zip(r['offered'], estimates, probabilities):
                    if a not in known:
                        self.assertIsNone(e)
                        self.assertEqual(probability, 0)
                    else:
                        expected = .75 * (.2 / len(available) + (.8 / ties if e == maximum else 0))
                        self.assertAlmostEqual(probability, expected)
                if r['round'] < c.rounds:
                    after = set(known)
                    if r['learned'][p] is not None:
                        after.add(r['learned'][p])
                    self.assertEqual(after, set(result['rounds'][r['round']]['knowledge_before'][p]))

    def test_question_table_includes_full_bank_and_probabilities(self):
        from GameSimulator.krillion.reporting import question_cells
        result = simulate(Config(players=2, rounds=2, absence_probability=.2, blank_probability=.3))
        cells = question_cells(result, 1, 1)
        r = result['rounds'][1]
        self.assertEqual(len(cells[0]), len(set(r['knowledge_before'][1]) | set(r['offered'])) + 2)
        self.assertEqual(cells[0][-2:], ['Blank', 'Absent'])
        self.assertEqual(cells[4][-2], '30.00%')
        self.assertEqual(cells[5][-2:], ['24.00%', '20.00%'])


if __name__ == '__main__':
    unittest.main()
