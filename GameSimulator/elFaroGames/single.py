"""One adaptive attendance predictor per player; deterministic decisions."""
from dataclasses import dataclass
import numpy as np


@dataclass
class SingleStrategyResult:
    forecasts: np.ndarray  # Expected OTHER attendees, before each decision
    attendance: np.ndarray
    rewards: np.ndarray
    capacity: int
    impulses: np.ndarray | None = None
    decision_forecasts: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    impulse_max: float | np.ndarray = 0.0


def simulate(num_players=100, num_rounds=1000, capacity=None,
             learning_rate=0.1, initial_forecast=None, seed=42, *, impulse_max=0.0):
    """Attend iff predicted other attendees + 1 <= capacity.

    Each player keeps a single exponentially weighted attendance predictor.
    After simultaneous choices, update it toward observed OTHER attendance.
    With impulse_max=0, randomness only initializes beliefs. Otherwise each
    decision subtracts an independent Uniform(0, impulse_max) impulse from
    the forecast, clipped at zero. The stored learned forecast is not reduced.
    impulse_max accepts a scalar or one nonnegative limit per player; zero
    gives that player no impulse. Limits stay fixed throughout the simulation.
    initial_forecast accepts a scalar or one expected other-attendee count
    per player. By default beliefs are uniform on [0, num_players - 1].
    """
    for name, value, minimum in (("num_players", num_players, 1),
                                 ("num_rounds", num_rounds, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if capacity is None:
        capacity = int(0.6 * num_players)
    if isinstance(capacity, bool) or not isinstance(capacity, (int, np.integer)) or not 0 <= capacity <= num_players:
        raise ValueError("capacity must be an integer between 0 and num_players")
    if not np.isfinite(learning_rate) or not 0 <= learning_rate <= 1:
        raise ValueError("learning_rate must be between 0 and 1")
    limits = np.asarray(impulse_max, dtype=float)
    scalar_limit = limits.ndim == 0
    if scalar_limit:
        limits = np.full(num_players, limits.item())
    if limits.shape != (num_players,) or not np.all(np.isfinite(limits)) or np.any(limits < 0):
        raise ValueError("impulse_max must be a nonnegative finite scalar or one value per player")
    active_impulse = limits > 0
    rng = np.random.default_rng(seed)
    if initial_forecast is None:
        initial = rng.uniform(0, num_players - 1, num_players)
    else:
        initial = np.asarray(initial_forecast, dtype=float)
        if initial.ndim == 0:
            initial = np.full(num_players, initial.item())
    if initial.shape != (num_players,) or not np.all(np.isfinite(initial)) or np.any((initial < 0) | (initial > num_players - 1)):
        raise ValueError("initial_forecast must contain expected other-attendee counts between 0 and num_players - 1")
    forecasts = np.empty((num_rounds + 1, num_players))
    forecasts[0] = initial
    attendance = np.zeros((num_rounds, num_players), dtype=bool)
    rewards = np.zeros((num_rounds, num_players))
    impulses = np.zeros((num_rounds, num_players))
    decision_forecasts = np.empty((num_rounds, num_players))
    for t in range(num_rounds):
        if active_impulse.any():
            impulses[t] = rng.uniform(0, limits, num_players)
        decision_forecasts[t] = np.maximum(0, forecasts[t] - impulses[t])
        choices = decision_forecasts[t] + 1 <= capacity
        total = int(choices.sum())
        attendance[t] = choices
        rewards[t] = choices * (1.0 if total <= capacity else -1.0)
        observed_others = total - choices.astype(int)
        forecasts[t + 1] = (1 - learning_rate) * forecasts[t] + learning_rate * observed_others
    probabilities = (forecasts + 1 <= capacity).astype(float)
    if capacity > 0 and active_impulse.any():
        # P(impulse >= forecast + 1 - capacity), before drawing the impulse.
        with np.errstate(over="ignore"):
            probabilities[:, active_impulse] = np.clip(
                1 - (forecasts[:, active_impulse] + 1 - capacity) / limits[active_impulse], 0, 1)
    return SingleStrategyResult(forecasts, attendance, rewards, capacity,
                                impulses, decision_forecasts, probabilities,
                                float(limits[0]) if scalar_limit else limits.copy())


def plot_simulation(result, output="el_farol_single.html", player=1):
    """Write a report with a player selector and cumulative rewards."""
    from .reporting import plot_player_report
    title = "El Farol: one adaptive strategy per player"
    if np.any(np.asarray(result.impulse_max) > 0):
        title += " with random impulse"
    return plot_player_report(result, output, title, player=player)
