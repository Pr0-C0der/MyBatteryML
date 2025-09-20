#!/usr/bin/env python3
import sys
from pathlib import Path
import itertools
import pandas as pd

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from batteryml.pipeline import Pipeline


def find_config_paths() -> list[Path]:
    base = ROOT / 'configs' / 'baselines'
    cfgs = []
    for sub in base.rglob('*.yaml'):
        cfgs.append(sub)
    return sorted(cfgs)


def evaluate_all(seed: int = 0, device: str = 'cpu') -> pd.DataFrame:
    rows = []
    for cfg in find_config_paths():
        try:
            pipe = Pipeline(config_path=str(cfg), workspace=None)
            # Train if needed (skip_if_executed True)
            pipe.train(seed=seed, device=device, skip_if_executed=True)
            # Evaluate and capture scores
            dataset, _ = None, None
            # Rebuild dataset inside evaluate; prediction and scores printed inside
            pipe.evaluate(seed=seed, device=device, metric=['RMSE', 'MAE'], skip_if_executed=False)
            # Collect last scores from workspace prediction file is heavier; recompute scores directly
            # Build dataset and model explicitly to get scores here
            model, dataset = pipe.train(seed=seed, device=device, skip_if_executed=True)
            prediction = model.predict(dataset)
            scores = {
                'RMSE': dataset.evaluate(prediction, 'RMSE'),
                'MAE': dataset.evaluate(prediction, 'MAE'),
            }
            rows.append({
                'config': str(cfg.relative_to(ROOT)),
                'workspace': str(pipe.config['workspace']) if pipe.config['workspace'] else '',
                **scores
            })
        except Exception as e:
            rows.append({
                'config': str(cfg.relative_to(ROOT)),
                'workspace': '',
                'RMSE': float('nan'),
                'MAE': float('nan'),
                'error': str(e)
            })
    return pd.DataFrame(rows)


def main():
    out_dir = ROOT / 'results'
    out_dir.mkdir(exist_ok=True)
    df = evaluate_all()
    csv_path = out_dir / 'baseline_rul_results.csv'
    md_path = out_dir / 'baseline_rul_results.md'
    df.to_csv(csv_path, index=False)
    with open(md_path, 'w') as f:
        f.write(df.to_markdown(index=False))
    print(f'Saved results to {csv_path} and {md_path}')


if __name__ == '__main__':
    main()


