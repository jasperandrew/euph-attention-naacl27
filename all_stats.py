"""Run selected model-level statistics workflows in one process."""
import argparse
import logging
from pathlib import Path

from src.stats.runner import (
    AVAILABLE_STEPS, discover_model_runs, run_many, run_selected,
)
from src.cli import configure_logging

logger = logging.getLogger(__name__)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description='Run selected statistics workflows, reusing loaded models.'
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--model-dir',
                           help='Run directory containing model/, or the model directory itself.')
    selection.add_argument('--all-models', action='store_true',
                           help='Run the selected workflows for every saved model under --runs-dir.')
    parser.add_argument('--runs-dir', type=Path, default=root / 'runs',
                        help='Model-run root used with --all-models (default: runs/).')
    parser.add_argument('--only', nargs='+', choices=AVAILABLE_STEPS, default=None,
                        help='Selected workflow(s); default runs all available workflows.')
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Output directory for a single model run.')
    parser.add_argument('--output-root', type=Path, default=None,
                        help='Output root for --all-models (default: output/stats/).')
    parser.add_argument('--jobs', type=int, default=1,
                        help='Number of model runs to process in parallel (default: 1).')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--skip-base', action='store_true',
                        help='Skip the base-model pass for significance tests.')
    parser.add_argument('--n-perms', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed per-metric statistical reports.')
    args = parser.parse_args()
    configure_logging(args.verbose)
    if args.jobs < 1:
        parser.error('--jobs must be at least 1.')

    steps = args.only or AVAILABLE_STEPS
    if args.all_models:
        if args.output_dir is not None:
            parser.error('--output-dir is for --model-dir; use --output-root with --all-models.')
        model_runs = discover_model_runs(args.runs_dir)
        if not model_runs:
            parser.error(f'No saved models found under {args.runs_dir}.')
        logger.info('Found %s model run(s) under %s.', len(model_runs), args.runs_dir)
        output_dirs = run_many(
            model_runs,
            workers=args.jobs,
            output_root=args.output_root,
            steps=steps,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            skip_base=args.skip_base,
            n_perms=args.n_perms,
            seed=args.seed,
            verbose=args.verbose,
        )
        logger.info('Completed statistics for %s model run(s).', len(output_dirs))
    else:
        if args.jobs != 1:
            parser.error('--jobs > 1 is only available with --all-models.')
        if args.output_root is not None:
            parser.error('--output-root is for --all-models; use --output-dir with --model-dir.')
        output_dir = run_selected(
            args.model_dir,
            steps=steps,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            skip_base=args.skip_base,
            n_perms=args.n_perms,
            seed=args.seed,
            verbose=args.verbose,
        )
        logger.info('Statistics written to %s', output_dir)


if __name__ == "__main__":
    main()
