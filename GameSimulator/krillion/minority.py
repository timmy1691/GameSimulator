"""Minority choice game with private experience and reusable answers."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class Config:
    players: int = 50
    rounds: int = 200
    answer_pool: int = 20
    answers_per_round: int = 6
    initial_knowledge: int = 10
    absence_probability: float = 0.2
    blank_probability: float = 0.0
    learning_probability: float = 1
    learning_rate: float = 0.5
    exploration: float = 0.1
    seed: int = 42

    def validate(self):
        for name in ('players', 'rounds', 'answer_pool', 'answers_per_round', 'initial_knowledge'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not 1 <= self.initial_knowledge <= self.answer_pool:
            raise ValueError('initial_knowledge must be <= answer_pool')
        if not 1 <= self.answers_per_round <= self.answer_pool:
            raise ValueError('answers_per_round must be <= answer_pool')
        for name in ('absence_probability', 'blank_probability', 'learning_probability', 'learning_rate', 'exploration'):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be finite and between 0 and 1')


def simulate(config=None):
    config = config or Config()
    config.validate()
    rng = np.random.default_rng(config.seed)
    # A shared fallback ensures every player can answer every generated question.
    # shared fallback being the value of 0
    known = [{0, *map(int, rng.choice(np.arange(1, config.answer_pool),
                                    config.initial_knowledge - 1, replace=False))}
             for _ in range(config.players)]
    
    initial = [sorted(k) for k in known]
    estimates = np.ones((config.players, config.answer_pool))
    histories = [[] for _ in known]
    totals = np.zeros(config.players)
    records = []
    for t in range(config.rounds):
        offered = sorted([0, *map(int, rng.choice(np.arange(1, config.answer_pool),
                         config.answers_per_round - 1, replace=False))])
        knowledge_before = [sorted(k) for k in known]
        answer_estimates = [[float(estimates[p, a]) if a in known[p] else None
                             for a in offered] for p in range(config.players)]
        answer_probabilities = []
        for p in range(config.players):
            available = sorted(known[p].intersection(offered))
            maximum = max(estimates[p, a] for a in available)
            best_known = [a for a in available if estimates[p, a] == maximum]
            answer_probabilities.append([
                (1 - config.blank_probability) * (
                    config.exploration / len(available)
                    + ((1 - config.exploration) / len(best_known) if a in best_known else 0))
                if a in known[p] else 0.0 for a in offered])
        present = rng.random(config.players) >= config.absence_probability
        choices = [None] * config.players
        for p in np.flatnonzero(present):
            available = sorted(known[p].intersection(offered))
            if rng.random() < config.blank_probability:
                continue
            values = estimates[p, available]
            candidates = available if rng.random() < config.exploration else [
                a for a, v in zip(available, values) if v == values.max()]
            choices[p] = int(rng.choice(candidates))
        counts = {a: choices.count(a) for a in offered}
        population = int(present.sum())
        payoffs = {a: population / n for a, n in counts.items() if n}
        best = [a for a, r in payoffs.items() if r == max(payoffs.values())] if payoffs else []
        rewards = [payoffs.get(a, 0.0) for a in choices]
        learned = [None] * config.players
        for p in np.flatnonzero(present):
            a = choices[p]
            if a is not None:
                estimates[p, a] += config.learning_rate * (rewards[p] - estimates[p, a])
            if best and rng.random() < config.learning_probability:
                winner = int(rng.choice(best))
                if winner not in known[p]:
                    learned[p] = winner
                known[p].add(winner)
                if winner != a:
                    estimates[p, winner] += config.learning_rate * (payoffs[winner] - estimates[p, winner])
            histories[p].append(dict(round=t + 1, choice=a, reward=rewards[p], learned=learned[p]))
        totals += rewards
        records.append(dict(round=t + 1,
            question=f'Round {t + 1}: which available route will be least crowded?',
            offered=offered, present=present.tolist(), choices=choices,
            knowledge_before=knowledge_before, answer_estimates=answer_estimates,
            answer_probabilities=answer_probabilities,
            population=population, counts=counts, best=best, rewards=rewards,
            learned=learned, knowledge_sizes=[len(k) for k in known], cumulative=totals.tolist()))
    return dict(config=asdict(config), initial_knowledge=initial, rounds=records,
                histories=histories, final_knowledge=[sorted(k) for k in known], totals=totals.tolist())


def write_report(result, output, player=1):
    from .reporting import write_report as report
    return report(result, output, player=player)
