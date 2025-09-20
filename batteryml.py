#!/usr/bin/env python

# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import os
import csv
import argparse

from pathlib import Path

from batteryml.preprocess import (
    DOWNLOAD_LINKS, download_file, SUPPORTED_SOURCES
)
from batteryml.pipeline import Pipeline
from batteryml.builders import PREPROCESSORS


def main():
    parser = argparse.ArgumentParser('BatteryML command line utilities.')
    subparsers = parser.add_subparsers()

    # download command
    download_parser = subparsers.add_parser(
        "download", help="Download raw files for public datasets")
    download_parser.add_argument(
        "dataset", choices=list(DOWNLOAD_LINKS.keys()),
        help="Public dataset to download")
    download_parser.add_argument(
        "output_dir", help="Directory to save the raw data files")
    download_parser.set_defaults(func=download)

    # preprocess command
    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Organize the raw data files into BatteryData and save to disk")
    preprocess_parser.add_argument(
        "input_type", choices=[value for values in SUPPORTED_SOURCES.values() for value in values],
        help="Type of input raw files. For public datasets, specific "
             "preprocessor will be called. For standard battery test "
             "output files, the corresponding preprocessing logic "
             "will be applied.")
    preprocess_parser.add_argument(
        "--config", default="None",
        help="Path to the config file of Cycler.")
    preprocess_parser.add_argument(
        "raw_dir", help="Directory of raw input files.")
    preprocess_parser.add_argument(
        "output_dir", help="Directory to save the BatteryData files.")
    preprocess_parser.add_argument(
        "-q", "--quiet", "--silent", dest="silent",
        action="store_true", help="Suppress logs during preprocessing.")
    preprocess_parser.set_defaults(func=preprocess)

    # run command
    run_parser = subparsers.add_parser(
        "run", help="Run the given config for training or evaluation")
    run_parser.add_argument(
        "config", help="Path to the config file")
    run_parser.add_argument(
        "--workspace", type=str, default=None, help="Directory to save the checkpoints and predictions.")
    run_parser.add_argument(
        "--device", default="cpu", help="Running device")
    run_parser.add_argument(
        "--ckpt-to-resume", "--ckpt_to_resume", dest="ckpt_to_resume",
        help="path to the checkpoint to resume training or evaluation")
    run_parser.add_argument(
        "--train", action="store_true",
        help="Run training. Will skip training if this flag is not provided.")
    run_parser.add_argument(
        "--eval", action="store_true",
        help="Run evaluation. Will skip eval if this flag is not provided.")
    run_parser.add_argument(
        "--metric", default="RMSE,MAE,MAPE",
        help="Metrics for evaluation, seperated by comma")
    run_parser.add_argument(
        "--seed", type=int, default=0, help="random seed")
    run_parser.add_argument(
        "--epochs", type=int, help="number of epochs override")
    run_parser.add_argument(
        "--skip_if_executed", type=str, default='False', help="skip train/evaluate if the model executed")
    run_parser.set_defaults(func=run)

    args = parser.parse_args()
    args.func(args)


def download(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    raw_dir = Path(args.output_dir)
    for f in DOWNLOAD_LINKS[args.dataset]:
        if len(f) == 2:
            (url, filename), total_length = f, None
        else:
            url, filename, total_length = f
        download_file(url, raw_dir / filename, total_length=total_length)


def preprocess(args):
    assert os.path.exists(
        args.raw_dir), f'Input path not exist: {args.raw_dir}'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    config_path = Path(args.config)
    input_path, output_path = Path(args.raw_dir), Path(args.output_dir)
    processor = PREPROCESSORS.build(dict(
        name=f'{args.input_type}Preprocessor',
        output_dir=output_path,
        silent=args.silent
    ))
    processor(input_path, config_path=config_path)


def run(args):
    # Convert skip_if_executed to boolean
    args.skip_if_executed = args.skip_if_executed.lower() in ['true', '1', 'yes']
    pipeline = Pipeline(args.config, args.workspace)
    model, dataset = None, None  # Reuse to save setup cost
    if args.train:
        model, dataset = pipeline.train(
            seed=args.seed,
            epochs=args.epochs,
            device=args.device,
            ckpt_to_resume=args.ckpt_to_resume,
            dataset=dataset,
            skip_if_executed=args.skip_if_executed)
    if args.eval:
        metric = args.metric.split(',')
        pipeline.evaluate(
            seed=args.seed,
            device=args.device,
            metric=metric,
            model=model,
            dataset=dataset,
            ckpt_to_resume=args.ckpt_to_resume,
            skip_if_executed=args.skip_if_executed
        )
        # After evaluation, log metrics into CSVs
        try:
            workspace = Path(args.workspace) if args.workspace is not None else None
            if workspace is not None and workspace.exists():
                # Collect all predictions in the workspace
                preds = sorted(workspace.glob('predictions_seed_*.pkl'))
                # Attempt to load all scores for aggregation
                all_scores = []
                for p in preds:
                    try:
                        import pickle
                        with open(p, 'rb') as f:
                            obj = pickle.load(f)
                            if 'scores' in obj:
                                all_scores.append((p.stem, obj['scores']))
                    except Exception:
                        pass
                # Log per-run metrics row
                per_run_csv = workspace / 'metrics.csv'
                per_run_csv_exists = per_run_csv.exists()
                with open(per_run_csv, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['config','seed','RMSE','MAE','MAPE'])
                    if not per_run_csv_exists:
                        writer.writeheader()
                    # If we have current scores printed in evaluation, fetch the latest file
                    if len(all_scores):
                        name, scores = all_scores[-1]
                        # extract seed from filename if possible
                        try:
                            seed_str = name.split('predictions_seed_')[1].split('_')[0]
                        except Exception:
                            seed_str = str(args.seed)
                        row = {
                            'config': str(args.config),
                            'seed': seed_str,
                            'RMSE': scores.get('RMSE'),
                            'MAE': scores.get('MAE'),
                            'MAPE': scores.get('MAPE'),
                        }
                        writer.writerow(row)

                # If this is a deep learning config (multiple seeds), log mean/std
                is_nn = 'nn_models' in str(args.config)
                if is_nn and len(all_scores) > 0:
                    import numpy as np
                    def agg(metric_key):
                        vals = [s.get(metric_key) for _, s in all_scores if metric_key in s]
                        if len(vals) == 0:
                            return None, None, 0
                        return float(np.mean(vals)), float(np.std(vals)), len(vals)

                    rmse_mean, rmse_std, n_rmse = agg('RMSE')
                    mae_mean, mae_std, n_mae = agg('MAE')
                    mape_mean, mape_std, n_mape = agg('MAPE')

                    agg_csv = workspace / 'metrics_aggregated.csv'
                    agg_exists = agg_csv.exists()
                    with open(agg_csv, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=[
                            'config','num_seeds',
                            'RMSE_mean','RMSE_std',
                            'MAE_mean','MAE_std',
                            'MAPE_mean','MAPE_std'
                        ])
                        if not agg_exists:
                            writer.writeheader()
                        writer.writerow({
                            'config': str(args.config),
                            'num_seeds': max(n_rmse, n_mae, n_mape),
                            'RMSE_mean': rmse_mean,
                            'RMSE_std': rmse_std,
                            'MAE_mean': mae_mean,
                            'MAE_std': mae_std,
                            'MAPE_mean': mape_mean,
                            'MAPE_std': mape_std,
                        })
        except Exception:
            # Logging should not break the primary flow
            pass


if __name__ == "__main__":
    main()
