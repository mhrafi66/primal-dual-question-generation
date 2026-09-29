#!/usr/bin/env python3
"""Run the four loss ablations for the restored primal-dual trainer.

The trainer is configured through PDQG_* environment variables so the historical
model code remains recognizable while experiments become reproducible.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(ROOT/'configs'/'ablation_matrix.json'))
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-train-samples', type=int, default=500)
    p.add_argument('--max-eval-samples', type=int, default=100)
    p.add_argument('--learning-rate', type=float, default=3e-5)
    p.add_argument('--only', nargs='*', help='Optional run names to execute')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()

def main():
    args = parse_args()
    configs = json.loads(Path(args.config).read_text())
    selected = set(args.only or [])
    log_dir = ROOT/'artifacts'/'experiments'
    log_dir.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        name = cfg['name']
        if selected and name not in selected:
            continue
        env = os.environ.copy()
        env.update({
            'PDQG_RUN_NAME': name,
            'PDQG_ALPHA': str(cfg['alpha']),
            'PDQG_BETA': str(cfg['beta']),
            'PDQG_EPOCHS': str(args.epochs),
            'PDQG_BATCH_SIZE': str(args.batch_size),
            'PDQG_MAX_TRAIN_SAMPLES': str(args.max_train_samples),
            'PDQG_MAX_EVAL_SAMPLES': str(args.max_eval_samples),
            'PDQG_LR': str(args.learning_rate),
            'PDQG_CHECKPOINT': str(ROOT/'checkpoints'/f'{name}.pth'),
        })
        cmd = [sys.executable, str(ROOT/'QuestionAnsweringTraining.py')]
        print(f"\n=== {name}: alpha={cfg['alpha']} beta={cfg['beta']} ===")
        print(' '.join(cmd))
        if args.dry_run:
            continue
        log_path = log_dir/f'{name}.log'
        with log_path.open('w') as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'log: {log_path}')

if __name__ == '__main__':
    main()
