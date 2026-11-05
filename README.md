# El Farol strategy simulator

## Different impulse limits for different players

Mix players with no impulse and players with different impulse strengths:

```sh
python main.py --strategy single-impulse --players 100 --rounds 1000 --capacity 60 --impulse-groups "0:50,2:25,5:25"
```

Players 1-50 have no impulse, players 51-75 draw impulses from Uniform(0, 2),
and players 76-100 draw from Uniform(0, 5). Group counts must sum to the player
count. Limits are fixed for the whole simulation; impulses are independently
drawn each round. Everyone still updates their forecast from actual attendance,
so even players without impulses can change decisions as attendance changes.

Alternatively, give one limit per player in player order:

```sh
python main.py --strategy single-impulse --players 4 --rounds 500 --capacity 2 --impulse-max-per-player "0,0,2,5"
```

Both options override the uniform `--impulse-max` value and require
`--strategy single-impulse`. Limits must be finite and nonnegative. Reports
default to `el_farol_heterogeneous_impulse.html`, preserving the uniform-model
report. The dropdown and title display the selected player's impulse limit,
alongside their forecast, attendance probability, decisions, and cumulative
reward. Use `--player 76` to start with a player in the highest-impulse group.

```python
from GameSimulator.elFaroGames import single

result = single.simulate(num_players=100, num_rounds=1000, capacity=60,
                         impulse_max=[0] * 50 + [2] * 25 + [5] * 25)
single.plot_simulation(result, 'el_farol_heterogeneous_impulse.html')
```

The probability formula uses each player's own limit. Zero-limit players have
probability exactly 0 or 1, without division by zero. A scalar limit still
applies equally to everyone; equal-valued lists produce the same seeded run.

## Player-focused reports and cumulative rewards

All three modes now show **one selected player at a time**. Use the dropdown
above the plots to switch to any player. Choose the initial player with
`--player 7` (player numbers start at 1). For example:

```sh
python main.py --strategy single-impulse --players 100 --rounds 1000 --capacity 60 --impulse-max 5 --player 7
```

The report shows the selected player's forecasts (single modes), attendance
probability, attend/stay-home decisions, and cumulative reward. Cumulative reward
starts at 0 in round 0; the point at round t includes rewards from rounds 1 to t.
Total attendance across all players remains visible for context. The dropdown
updates all player-specific charts together; there are no overlapping curves
from other players. In impulse mode, the selected player's learned and adjusted
forecasts are shown together so the impulse effect is visible.

Python: `plot_simulation(result, 'report.html', player=7)`.

Run current tests with:
`python -m unittest test_mixed test_single test_impulse test_reporting -v`.


## Single strategy with a random impulse

```sh
python main.py --strategy single-impulse --players 100 --rounds 1000 --capacity 60 --impulse-max 5
```

This additional mode writes `el_farol_single_impulse.html`. Each round, each
player independently draws an impulse uniformly between 0 and `impulse_max`,
measured in attendee counts. It applies from the first round onward:

```text
decision forecast = max(0, learned forecast of others - impulse)
attend if decision forecast + 1 <= capacity
```

The impulse is temporary. The learned forecast still updates toward actual
other-player attendance using the original exponential moving average. Rewards
remain +1 / -1 / 0. The seed controls initial beliefs and all impulse draws.
`--impulse-max 0` exactly reproduces single mode; negative/nonfinite values are
rejected. The default maximum is 5 attendees, independent of population size.

The report includes learned forecasts (solid), temporary decision forecasts
(dotted), decisions, total attendance, and the selected player's attendance probability and cumulative reward. For positive
capacity and positive impulse maximum M, before drawing the impulse:

`P(attend) = clip(1 - (forecast + 1 - capacity) / M, 0, 1)`

Capacity zero always gives probability zero because a player needs their own
seat. The last probability column is for the next unplayed round. These are
probabilities induced by noise in one forecasting rule, rather than the mixed
model's softmax payoff learning.

Convergence to permanent groups is not universal. Impulses can trigger extra
attendance and subsequent forecast changes, but bounded impulses may be too
small to change a player's decision. In an exact at-capacity split with fully
settled forecasts, outsiders need a deduction of at least one attendee to
join; a continuous uniform impulse with maximum at most one has zero
probability of crossing that threshold. Larger impulses can disrupt that split.

Python API: `single.simulate(..., impulse_max=5)` returns `impulses` and
`decision_forecasts` with shape (rounds, players), plus `probabilities` with
shape (rounds + 1, players), in addition to forecasts, attendance, and rewards.

Run all current tests: `python -m unittest test_mixed test_single test_impulse -v`.

## Single adaptive strategy per player

```sh
python main.py --strategy single --players 100 --rounds 1000 --capacity 60
```

Open `el_farol_single.html` for the selected player's forecast, probability,
decisions, cumulative reward, and total attendance. The existing mixed
model remains the default (`--strategy mixed`). The two reports use separate
default filenames.

In single mode, every player keeps one forecasting rule: an exponentially
weighted average of historical **other-player attendance**. They attend when
their forecast plus their own seat is at most capacity. All decisions happen
simultaneously, before that round's attendance is known. Afterward:

`forecast_next = (1 - learning_rate) * forecast + learning_rate * observed_others`

The payoff is the same as mixed mode: +1 for successful attendance, -1 for
overcrowded attendance, and 0 for staying home. Each player adapts their own
forecast; there is no switching among strategy algorithms or randomized action
selection. Conditional on their forecast, their attendance probability is 0 or
1, so the report shows forecasts and actual decisions instead of smooth
probability curves. Synchronized choices and attendance cycles are possible.

Initial forecasts are seeded random values between 0 and players - 1, providing
different starting beliefs. `--initial-forecast 30` gives everyone the same
initial belief about other attendees. `--learning-rate` controls adaptation;
zero freezes beliefs. `--temperature` and `--initial-probability` apply only to
mixed mode; `--initial-forecast` applies only to single mode.

```python
from GameSimulator.elFaroGames import single

result = single.simulate(num_players=3, num_rounds=100, capacity=2,
                         initial_forecast=[0, 1, 2], learning_rate=0.1)
print(result.forecasts)  # (101, 3), expected OTHER attendees
print(result.attendance)  # (100, 3), Boolean decisions
single.plot_simulation(result, 'single_example.html')
```

The plotted forecast includes the player's own seat so it can be compared
directly with capacity. The final forecast is for the next unplayed round.
Run both current model test suites with:
`python -m unittest test_mixed test_single -v`.

## Mixed strategies

Python 3.10 or later. Install dependencies and run:

```sh
python -m pip install -r requirements.txt
python main.py --players 100 --rounds 1000 --capacity 60 --seed 42
```

Open `el_farol.html` in your browser. It works offline and includes:
- The selected player's probability, decisions, and cumulative reward.
- A dropdown to inspect any other player. Hover for exact values and zoom.
- Actual total attendance with the capacity marked.

## Model

A mixed strategy assigns probabilities to the two actions: attend with p, stay
home with 1-p. Every player independently samples their action before anyone
sees that round's attendance. This implements adaptive mixing over actions,
not a portfolio of different attendance forecasting algorithms.

Attending earns +1 when total attendance is at most capacity and -1 otherwise.
Staying home earns 0. The default capacity is floor(0.6 * players). These are
standard fixed El Farol payoffs; the older routines' attendance-scaled rewards
are not used by the new simulator.

After each round, players observe total attendance and update their estimated
attendance payoff Q using Q <- Q + learning_rate * (hypothetical payoff - Q).
For a player who stayed home, hypothetical attendance includes their extra seat.
The stay-home payoff is known to be zero. The next attendance probability is
p = sigmoid(Q / temperature). This uses historical public information only.

All players start at p=0.5 by default. Equal probabilities can overlap, especially
away from the capacity boundary. Players can diverge at the boundary because
attending and staying home give different hypothetical outcomes. This is a
learning simulation, not a guarantee of convergence to a Nash equilibrium.

Options:
- `--players`, `--rounds`: population and number of rounds (zero rounds allowed).
- `--capacity`: inclusive attendance limit, from zero to population size.
- `--learning-rate`: 0 to 1; higher values emphasize recent outcomes.
- `--temperature`: positive; higher values keep probabilities nearer 0.5.
- `--initial-probability`: strictly between 0 and 1.
- `--seed`: repeatable random draws.
- `--output`: HTML output path, default `el_farol.html`.

## Python use

```python
from GameSimulator.elFaroGames.mixed import simulate, plot_simulation

result = simulate(num_players=50, num_rounds=500, capacity=30, seed=7)
# Probability used by player 1 for round 1:
print(result.probabilities[0, 0])
# All players' probability trajectories:
print(result.probabilities.shape)  # (501, 50)
print(result.rewards.sum(axis=0))  # cumulative score per player
plot_simulation(result, 'my_simulation.html')
```

`initial_probability` also accepts one value per player in the Python API.
Rows of `probabilities` are decision rounds; its final row is the probability
for the next unplayed round. Attendance and rewards have shape (rounds, players).

Run new simulator tests with `python -m unittest test_mixed -v`.
The existing `test_game.py` belongs to an older interface and imports a missing
`game.py`; it cannot run against this repository as supplied.
