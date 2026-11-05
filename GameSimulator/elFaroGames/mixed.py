"""Adaptive mixed strategies for repeated El Farol play."""
from dataclasses import dataclass
import numpy as np

@dataclass
class SimulationResult:
    probabilities: np.ndarray  # (rounds + 1, players): initial through final update
    attendance: np.ndarray
    rewards: np.ndarray
    capacity: int


def simulate(num_players=100, num_rounds=1000, capacity=None,
             learning_rate=0.1, temperature=0.25, initial_probability=0.5, seed=42):
    """Softmax mixing over attend/stay home, learning from public attendance.

    Attendance pays +1 at or below capacity, -1 above it; staying home pays 0.
    Each player learns the hypothetical attendance payoff, counting their own
    extra seat when they stayed home. Decisions use only completed rounds.
    """
    for name, value, minimum in (("num_players", num_players, 1), ("num_rounds", num_rounds, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if capacity is None:
        capacity = int(0.6 * num_players)
    if isinstance(capacity, bool) or not isinstance(capacity, (int, np.integer)) or not 0 <= capacity <= num_players:
        raise ValueError("capacity must be an integer between 0 and num_players")
    if not np.isfinite(learning_rate) or not 0 <= learning_rate <= 1:
        raise ValueError("learning_rate must be between 0 and 1")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    initial = np.asarray(initial_probability, dtype=float)
    if initial.ndim == 0:
        initial = np.full(num_players, initial.item())
    if initial.shape != (num_players,) or not np.all(np.isfinite(initial)) or np.any((initial <= 0) | (initial >= 1)):
        raise ValueError("initial_probability must be a scalar or one value per player, strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    probabilities = np.empty((num_rounds + 1, num_players))
    attendance = np.zeros((num_rounds, num_players), dtype=bool)
    rewards = np.zeros((num_rounds, num_players))
    probabilities[0] = initial
    estimates = temperature * (np.log(initial) - np.log1p(-initial))
    for t in range(num_rounds):
        choices = rng.random(num_players) < probabilities[t]
        total = int(choices.sum())
        attendance[t] = choices
        rewards[t] = choices * (1.0 if total <= capacity else -1.0)
        hypothetical_total = total + (~choices).astype(int)
        attend_payoff = np.where(hypothetical_total <= capacity, 1.0, -1.0)
        estimates += learning_rate * (attend_payoff - estimates)
        with np.errstate(over="ignore"):
            probabilities[t + 1] = np.exp(-np.logaddexp(0, -estimates / temperature))
    return SimulationResult(probabilities, attendance, rewards, capacity)


def plot_simulation(result, output="el_farol.html", player=1):
    """Write a report with a player selector and cumulative rewards."""
    from .reporting import plot_player_report
    title = "El Farol: adaptive mixed strategies"
    return plot_player_report(result, output, title, player=player)
