#!/usr/bin/env python3
import sys
from pathlib import Path
import pickle
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


def _workspace_for_config(cfg: Path) -> Path:
    # Mirror batteryml.pipeline default but under a dedicated 'workspaces/eval' root
    rel = cfg.relative_to(ROOT / 'configs').with_suffix('')
    return ROOT / 'workspaces' / 'eval' / rel


def _latest_scores(workspace: Path) -> dict:
    preds = sorted(workspace.glob('predictions_seed_*.pkl'))
    if not preds:
        return {}
    with open(preds[-1], 'rb') as f:
        obj = pickle.load(f)
    return obj.get('scores', {})


def evaluate_all(seed: int = 0, device: str = 'cpu') -> pd.DataFrame:
    rows = []
    for cfg in find_config_paths():
        try:
            ws = _workspace_for_config(cfg)
            ws.mkdir(parents=True, exist_ok=True)

            # Emulate CLI: build pipeline, train, then eval
            pipe = Pipeline(str(cfg), str(ws))
            model, dataset = pipe.train(
                seed=seed,
                device=device,
                ckpt_to_resume=None,
                skip_if_executed=False
            )
            pipe.evaluate(
                seed=seed,
                device=device,
                metric=['RMSE','MAE','MAPE'],
                model=model,
                dataset=dataset,
                ckpt_to_resume=None,
                skip_if_executed=False
            )

            scores = _latest_scores(ws)
            rows.append({
                'config': str(cfg.relative_to(ROOT)),
                'workspace': str(ws),
                'RMSE': scores.get('RMSE', float('nan')),
                'MAE': scores.get('MAE', float('nan')),
                'MAPE': scores.get('MAPE', float('nan')),
            })
        except Exception as e:
            rows.append({
                'config': str(cfg.relative_to(ROOT)),
                'workspace': '',
                'RMSE': float('nan'),
                'MAE': float('nan'),
                'MAPE': float('nan'),
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


