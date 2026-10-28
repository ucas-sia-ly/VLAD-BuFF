"""Run the README stages, keeping the original training protocol.

Launch through Trainroom using reproduction/run.sh. Each variant has an
independent directory; a failed stage stops the pipeline without overwriting it.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
HELPER = Path('/home/admin123/.codex/skills/local-training-terminal/scripts/trainroom.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['full', 'pca192'], required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--chain-192', action='store_true')
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if root.exists():
        raise FileExistsError(f'{root} already exists. Use a new experiment directory.')
    root.mkdir(parents=True)
    projection = [] if args.variant == 'full' else ['--nv_pca', '192']
    pcs = '8192' if args.variant == 'full' else '4096'
    name = 'dnv2_NV_AB' if args.variant == 'full' else 'dnv2_NV_192PCA_AB'
    common = ['--aggregation', 'NETVLAD', '--expName', name, '--antiburst', *projection]
    log_dir = root / 'lightning_logs' / 'version_0'
    checkpoint = log_dir / 'checkpoints' / 'last.ckpt'
    whitened = checkpoint.with_name(f'wpca{pcs}_last.ckpt')
    state = {'variant': args.variant, 'status': 'running', 'started': time.time(),
             'training_protocol': {'epochs': 4, 'places_per_batch': 60,
                                   'images_per_place': 4, 'resize': [224, 224],
                                   'precision': '16-mixed', 'seed': 1,
                                   'trainable_blocks': 4},
             'checkpoint': str(checkpoint), 'whitened_checkpoint': str(whitened),
             'stages': []}
    managed_dir = os.environ.get('TRAIN_TERMINAL_RUN_DIR')
    run_id = Path(managed_dir).name if managed_dir else None

    def save():
        temporary = root / 'pipeline.tmp'
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(root / 'pipeline.json')

    def stage(label, command, link=False):
        entry = {'name': label, 'command': command, 'started': time.time(), 'status': 'running'}
        state['stages'].append(entry)
        state['phase'] = label
        save()
        print(f'\n===== {name}: {label} =====\n{command}', flush=True)
        process = subprocess.Popen(command, cwd=PROJECT)
        linked = False
        while process.poll() is None:
            csv = log_dir / 'metrics.csv'
            if link and run_id and not linked and csv.is_file():
                result = subprocess.run([sys.executable, str(HELPER), 'link', '--id', run_id,
                                         '--path', str(csv)], check=False)
                linked = result.returncode == 0
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        entry.update(ended=time.time(), exit_code=process.returncode,
                     status='completed' if process.returncode == 0 else 'failed')
        save()
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)

    try:
        stage('train', [sys.executable, '-u', 'train.py', *common, '--no_wandb',
                       '--batch_size', '60', '--epochs', '4', '--num_workers', '4',
                       '--save_dir', str(root)], link=True)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        stage('whitening', [sys.executable, '-u', 'add_pca.py', *common,
                            '--ckpt_state_dict', '--num_pcs', pcs,
                            '--batch_size', '32', '--num_workers', '4',
                            '--resume_train', str(checkpoint),
                            '--features_path', str(root / 'pca_features.dat'), '--bounded_memory'])
        stage('MSLS evaluation', [sys.executable, '-u', 'eval.py', *common,
                                 '--wpca', '--num_pcs', pcs, '--ckpt_state_dict',
                                 '--val_datasets', 'MSLS', '--resume_train', str(whitened),
                                 '--batch_size', '32', '--num_workers', '4',
                                 '--store_eval_output', '--save_dir', str(root / 'evaluation'),
                                 '--no_wandb'])
        state.update(status='completed', ended=time.time())
        save()
        if args.chain_192 and args.variant == 'full':
            next_root = root.with_name(root.name + '_192')
            command = [sys.executable, str(HELPER), 'launch', '--cwd', str(PROJECT),
                       '--name', 'VLAD-BuFF 192PrePool · README reproduction',
                       '--total-steps', str(math.ceil(62514 / 60) * 4), '--',
                       '/usr/bin/bash', str(PROJECT / 'reproduction/run.sh'),
                       '--variant', 'pca192', '--run-dir', str(next_root)]
            # Never retry an ambiguous launch. The helper reports an acknowledged ID.
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            (root / 'next-run.json').write_text(result.stdout)
            print(result.stdout, flush=True)
            result.check_returncode()
    except BaseException as exc:
        state.update(status='failed', ended=time.time(), error=repr(exc))
        save()
        raise


if __name__ == '__main__':
    main()
