import argparse
from pathlib import Path
from GameSimulator.elFaroGames.mixed import simulate, plot_simulation
from GameSimulator.elFaroGames import single


def main():
    parser = argparse.ArgumentParser(description="El Farol with adaptive mixed or single strategies")
    parser.add_argument("--strategy", choices=("mixed", "single", "single-impulse"), default="mixed")
    parser.add_argument("--impulse-max", type=float, default=1.0,
                        help="Single-impulse mode: maximum forecast deduction, in attendees (default 5)")
    impulse_options = parser.add_mutually_exclusive_group()
    impulse_options.add_argument("--impulse-max-per-player",
                                 help="Comma-separated limits in player order, e.g. 0,0,2,5; overrides --impulse-max")
    impulse_options.add_argument("--impulse-groups",
                                 help="Limit:count groups, e.g. 0:50,2:25,5:25; overrides --impulse-max")
    parser.add_argument("--players", type=int, default=100)
    parser.add_argument("--player", type=int, default=1, help="Initially displayed player (1-based); change with report dropdown")
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--capacity", type=int, default=None, help="Default: floor(60%% of players)")
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--initial-probability", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-forecast", type=float, default=None,
                        help="Single strategy: initial expected OTHER attendees; default random beliefs")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if not 1 <= args.player <= args.players:
        parser.error("--player must be between 1 and --players")
    heterogeneous = args.impulse_max_per_player is not None or args.impulse_groups is not None
    if heterogeneous and args.strategy != "single-impulse":
        parser.error("Per-player impulse options require --strategy single-impulse")
    try:
        impulse_limits = args.impulse_max
        if args.impulse_max_per_player is not None:
            impulse_limits = [float(value) for value in args.impulse_max_per_player.split(',')]
        elif args.impulse_groups is not None:
            impulse_limits = []
            for group in args.impulse_groups.split(','):
                limit, count = group.split(':')
                count = int(count)
                if count < 1 or count > args.players - len(impulse_limits):
                    raise ValueError("Group counts must be positive and sum to --players")
                impulse_limits.extend([float(limit)] * count)
        if heterogeneous and len(impulse_limits) != args.players:
            raise ValueError("Provide exactly one impulse limit per player; group counts must sum to --players")
        if args.strategy in ("single", "single-impulse"):
            result = single.simulate(args.players, args.rounds, args.capacity,
                                     args.learning_rate, args.initial_forecast, args.seed,
                                     impulse_max=impulse_limits if args.strategy == "single-impulse" else 0)
        else:
            result = simulate(args.players, args.rounds, args.capacity, args.learning_rate,
                              args.temperature, args.initial_probability, args.seed)
    except ValueError as error:
        parser.error(str(error))
    filenames = {"mixed": "el_farol.html", "single": "el_farol_single.html",
                 "single-impulse": "el_farol_single_impulse.html"}
    output = args.output or filenames[args.strategy]
    if heterogeneous and args.output is None:
        output = "el_farol_heterogeneous_impulse.html"
    plotter = single.plot_simulation if args.strategy != "mixed" else plot_simulation
    plotter(result, output, player=args.player)
    print(f"Interactive report: {Path(output).resolve()}")
    if args.rounds:
        print(f"Mean attendance: {result.attendance.sum(axis=1).mean():.2f}; capacity: {result.capacity}")


if __name__ == "__main__":
    main()
