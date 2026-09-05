"""Command-line entry point for the Chopsticks solver (``chopsticks-solve``)."""

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, List

from chopsticks.env.state import ChopsticksState, Turn
from chopsticks.solver import CANONICAL_START, format_stats, graph_stats, solve

if TYPE_CHECKING:
    import pydot

_TURNS = {"P1": Turn.P1, "P2": Turn.P2}

# graph name -> output file stem
_GRAPHS = {
    "optimal": "optimal_graph",
    "p1-winning": "p1_winning_p2_all",
    "full": "full_graph",
}

# `full` is the brute-forced 1250-node graph; PNG rendering it takes minutes, so
# it is opt-in rather than part of the default set.
_DEFAULT_GRAPHS = ("optimal", "p1-winning")


def parse_state(text: str) -> ChopsticksState:
    """Parse ``"p1min,p1max,p2min,p2max,P1|P2"`` into a :class:`ChopsticksState`."""

    parts = [piece.strip() for piece in text.split(",")]
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            f"expected 'p1min,p1max,p2min,p2max,P1|P2', got {text!r}"
        )
    *hands, turn = parts
    if turn.upper() not in _TURNS:
        raise argparse.ArgumentTypeError(f"turn must be P1 or P2, got {turn!r}")
    try:
        p1_min, p1_max, p2_min, p2_max = (int(hand) for hand in hands)
    except ValueError:
        raise argparse.ArgumentTypeError(f"hand counts must be integers: {text!r}")
    return ChopsticksState(p1_min, p1_max, p2_min, p2_max, _TURNS[turn.upper()])


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for ``chopsticks-solve``."""

    parser = argparse.ArgumentParser(
        prog="chopsticks-solve",
        description="Solve Chopsticks by retrograde analysis and render the "
        "optimal-strategy graphs.",
    )
    parser.add_argument(
        "--start",
        type=parse_state,
        default=CANONICAL_START,
        metavar='"1,1,1,1,P1"',
        help="starting position (default: the canonical opening)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        metavar="PATH",
        help="directory for rendered graphs (default: current directory)",
    )
    parser.add_argument(
        "--graph",
        action="append",
        choices=sorted(_GRAPHS),
        dest="graphs",
        help="graph to render; repeatable (default: optimal + p1-winning; "
        "'full' is the slow brute-forced graph)",
    )
    parser.add_argument(
        "--format",
        choices=("dot", "png", "both"),
        default="both",
        help="output format(s) for rendered graphs (default: both)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="solve and print stats only; write no files",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="print stats as JSON instead of text",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the progress bar",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version('chopsticks')}",
    )
    return parser


def _formats(choice: str) -> List[str]:
    """Expand the ``--format`` choice into concrete formats."""

    return ["dot", "png"] if choice == "both" else [choice]


def main(argv: List[str] | None = None) -> int:
    """Solve the game, print stats, and optionally render graphs. Returns ``0``."""

    args = build_parser().parse_args(argv)

    quiet = args.quiet or args.as_json
    solution = solve(args.start, progress=not quiet)
    stats = graph_stats(solution, start=args.start)
    if args.as_json:
        print(json.dumps(stats))
    else:
        print(format_stats(stats))

    if args.no_render:
        return 0

    try:
        from chopsticks.solver import viz
    except ImportError:
        print(
            "graph rendering needs the 'viz' extra: uv sync --extra viz "
            "(or pip install 'chopsticks[viz]'); use --no-render to skip it.",
            file=sys.stderr,
        )
        return 1

    def build(name: str) -> "pydot.Dot":
        if name == "optimal":
            return viz.optimal_graph_dot(solution, args.start)
        if name == "p1-winning":
            return viz.p1_winning_dot(solution, args.start)
        return viz.full_state_graph_dot()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.graphs or _DEFAULT_GRAPHS:
        dot_graph = build(name)
        for fmt in _formats(args.format):
            path = args.output_dir / f"{_GRAPHS[name]}.{fmt}"
            written = viz.render(dot_graph, path, fmt)
            if written is not None:
                print(f"wrote {written}")
    return 0
