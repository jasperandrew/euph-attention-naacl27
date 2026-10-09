"""In-process orchestration for selected model statistics workflows."""

from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import multiprocessing
from pathlib import Path

from .. import config
from .models import ModelCache, use_model_cache


logger = logging.getLogger(__name__)


AVAILABLE_STEPS = ('evaluate', 'significance', 'sanity_checks')
EXECUTION_ORDER = ('evaluate', 'sanity_checks', 'significance')


def discover_model_runs(runs_dir):
    """Find run directories containing a saved model configuration."""
    runs_dir = Path(runs_dir).expanduser().resolve()
    if not runs_dir.is_dir():
        raise FileNotFoundError(f'Runs directory does not exist: {runs_dir}')
    return sorted(
        config_path.parent.parent
        for config_path in runs_dir.rglob('model/config.json')
    )


def run_selected(model_dir, steps=AVAILABLE_STEPS, output_dir=None, *,
                 max_samples=None, batch_size=32, skip_base=False,
                 n_perms=20, seed=0, verbose=False):
    """Run selected analyses in one process and reuse models and profiles.

    Sanity checks run before significance when both are selected, allowing
    significance tests to reuse the real-PET profiles from that pass.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(message)s', force=True,
    )
    unknown = set(steps) - set(AVAILABLE_STEPS)
    if unknown:
        raise ValueError(f'Unknown statistics step(s): {", ".join(sorted(unknown))}')
    selected_steps = [step for step in EXECUTION_ORDER if step in set(steps)]

    config.set_model_dir(model_dir)
    output_dir = Path(output_dir) if output_dir else (
        Path(config.BASE_DIR) / 'output' / 'stats' / config.MODEL_RUN_NAME
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import only selected workflows; tokenizer initialization is lazy.
    workflows = {}
    if 'evaluate' in selected_steps:
        import evaluate
        workflows['evaluate'] = (evaluate.main, {'batch_size': batch_size})
    if 'significance' in selected_steps:
        import significance
        workflows['significance'] = (significance.main, {'skip_base': skip_base})
    if 'sanity_checks' in selected_steps:
        import sanity_checks
        workflows['sanity_checks'] = (
            sanity_checks.main, {'n_perms': n_perms, 'seed': seed},
        )

    cache = ModelCache()
    shared_profiles = None
    try:
        with use_model_cache(cache):
            for step in selected_steps:
                logger.info('Running %s...', step)
                main, extra_args = workflows[step]
                args = Namespace(
                    model_dir=str(model_dir),
                    json_out=str(output_dir / f'{step}.json'),
                    max_samples=max_samples,
                    **extra_args,
                )
                if step == 'significance' and shared_profiles is not None:
                    result = main(args, precomputed_profiles=shared_profiles)
                else:
                    result = main(args)
                if isinstance(result, dict) and 'real_ft' in result:
                    shared_profiles = {
                        **(shared_profiles or {}),
                        **{key: value for key, value in result.items() if value is not None},
                    }

        if max_samples is None and shared_profiles is not None:
            plot_data_dir = Path(config.PLOTS_DIR) / config.MODEL_RUN_NAME / 'data'
            plot_data_dir.mkdir(parents=True, exist_ok=True)
            for key, filename in (
                ('real_ft', 'ft_df.parquet'),
                ('base_real', 'base_df.parquet'),
                ('masked_ft', 'mask_df.parquet'),
            ):
                dataframe = shared_profiles.get(key)
                if dataframe is not None:
                    dataframe.drop(columns=['sample_id'], errors='ignore').to_parquet(
                        plot_data_dir / filename, index=False,
                    )
    finally:
        cache.clear()

    return output_dir


def _run_model_job(job):
    """Process-pool entry point; each worker owns its model/cache state."""
    return str(run_selected(**job))


def run_many(model_dirs, *, workers=1, output_root=None, steps=AVAILABLE_STEPS,
             max_samples=None, batch_size=32, skip_base=False,
             n_perms=20, seed=0, verbose=False):
    """Run the selected workflows for multiple models, optionally in parallel."""
    if workers < 1:
        raise ValueError('workers must be at least 1')
    model_dirs = [Path(path).expanduser().resolve() for path in model_dirs]
    if not model_dirs:
        raise ValueError('No model runs were selected.')
    run_names = [path.name for path in model_dirs]
    if len(set(run_names)) != len(run_names):
        raise ValueError('Model run directory names must be unique for separate outputs.')

    output_root = Path(output_root).expanduser().resolve() if output_root else None
    jobs = []
    for model_dir in model_dirs:
        run_name = model_dir.name
        output_dir = output_root / run_name if output_root else None
        jobs.append({
            'model_dir': str(model_dir),
            'steps': tuple(steps),
            'output_dir': str(output_dir) if output_dir else None,
            'max_samples': max_samples,
            'batch_size': batch_size,
            'skip_base': skip_base,
            'n_perms': n_perms,
            'seed': seed,
            'verbose': verbose,
        })

    if workers == 1:
        return [_run_model_job(job) for job in jobs]

    failures = []
    completed = []
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        future_to_name = {
            executor.submit(_run_model_job, job): Path(job['model_dir']).name
            for job in jobs
        }
        for future in as_completed(future_to_name):
            run_name = future_to_name[future]
            try:
                output_dir = future.result()
                completed.append(output_dir)
                logger.info('Completed stats for %s: %s', run_name, output_dir)
            except Exception as exc:
                failures.append((run_name, exc))
                logger.exception('Failed stats for %s', run_name, exc_info=exc)
    if failures:
        failed_names = ', '.join(name for name, _ in failures)
        raise RuntimeError(f'Stats failed for {len(failures)} model run(s): {failed_names}')
    return completed
