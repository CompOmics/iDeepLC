"""iDeepLC: Deep learning-based retention time prediction."""

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

from ideeplc import __version__
from ideeplc.config import get_config
from ideeplc.ideeplc_core import main as run_ideeplc  # Assumes main logic is exposed here

LOG_MAPPING = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
LOGGER = logging.getLogger("ideeplc")
CONSOLE = Console(record=True)


def _print_credits():
    """Print package credits."""
    text = Text()
    text.append("\niDeepLC", style="bold link https://github.com/Alirezak2n/ideeplc")
    text.append(f" (v{__version__})\n", style="bold")
    text.append("Deep learning-based retention time prediction.\n")
    text.append("Developed at CompOmics.\n")
    text.append("Please cite: doi\n")
    text.stylize("cyan")
    CONSOLE.print(text)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="iDeepLC: Deep learning-based retention time prediction",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=42),
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the (CSV) file containing the peptide sequences.")
    parser.add_argument("--save_results", action="store_true",
                        help="Flag to save results to disk.")
    parser.add_argument("--log_level", type=str, default="info",
                        choices=LOG_MAPPING.keys(),
                        help="Logging level (default: info).")
    return parser


def _setup_logging(level: str, log_file: Path = None):
    if level not in LOG_MAPPING:
        raise ValueError(f"Invalid log level '{level}'. Choose from {', '.join(LOG_MAPPING)}")
    handlers = [RichHandler(rich_tracebacks=True, console=CONSOLE, show_path=False)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
    logging.basicConfig(
        format="%(name)s || %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=LOG_MAPPING[level],
        handlers=handlers,
    )


def main():
    _print_credits()

    parser = _argument_parser()
    args = parser.parse_args()

    # Optional log to file: ./ideeplc_run.log
    log_file = Path(f"ideeplc.log")
    _setup_logging(args.log_level, log_file=log_file)

    try:
        run_ideeplc(args)  # This should be the entry function from your main logic
    except Exception as e:
        LOGGER.exception("Execution failed.")
        sys.exit(1)



if __name__ == "__main__":
    main()
