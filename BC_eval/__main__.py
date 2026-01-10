"""Entry point for BC evaluation module.

Usage:
    uv run -m BC_eval --config BC_eval/configs/eval_margit.yaml
"""

import argparse
from pathlib import Path

from .evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained BC models in EldenGym",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate with a config file
    uv run -m BC_eval --config BC_eval/configs/eval_margit.yaml
    
    # Quick test with random policy
    uv run -m BC_eval --config BC_eval/configs/eval_random.yaml
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to evaluation config YAML file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override checkpoint path from config",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Override number of episodes from config",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=None,
        help="Enable rendering (overrides config)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering (overrides config)",
    )

    args = parser.parse_args()

    # Build overrides dict
    overrides = {}
    if args.checkpoint:
        overrides["checkpoint_path"] = args.checkpoint
    if args.num_episodes:
        overrides["num_episodes"] = args.num_episodes
    if args.render:
        overrides["render"] = True
    elif args.no_render:
        overrides["render"] = False

    # Run evaluation
    evaluate(args.config, overrides=overrides)


if __name__ == "__main__":
    main()


