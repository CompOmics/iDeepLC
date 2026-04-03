"""Build standardized modification features from a user CSV."""

import argparse
import logging

from ideeplc.utilities import build_user_mod_feature_table


LOGGER = logging.getLogger(__name__)


def _argument_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the feature builder."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert a CSV with columns name, aa, smiles into standardized modification features."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the CSV file containing user modifications.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="user_mod_features_standardized.csv",
        help="Output path for the standardized feature table.",
    )
    return parser


def main(argv=None):
    """Build standardized modification features from a raw user CSV."""
    parser = _argument_parser()
    args = parser.parse_args(argv)

    feature_table = build_user_mod_feature_table(args.input, args.output)
    LOGGER.info(
        "Wrote %d modification feature rows to %s", len(feature_table), args.output
    )


if __name__ == "__main__":
    main()
