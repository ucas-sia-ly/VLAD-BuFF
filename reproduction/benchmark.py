"""Adapt supplied datasets, then evaluate existing checkpoints under author protocols."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import hashlib
from functools import lru_cache
import shutil

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from reproduction.prepare_datasets import prepare, PAPER_DATASETS

PAPER_URL = 'https://arxiv.org/html/2409.19293v1#S5.T1'
PAPER = {
    'MSLS': [(92.4, 95.8), (91.9, 95.9)],
    'Nordland': [(78.0, 90.4), (71.4, 86.3)],
    'pitts250k_test': [(95.6, 98.7), (95.0, 98.2)],
    'SPED': [(92.8, 96.2), (90.9, 96.0)],
    'sfsm': [(88.3, 91.0), (87.3, 90.1)],
    'tokyo247': [(96.5, 98.1), (97.5, 98.4)],
    'st_lucia': [(100.0, 100.0), (99.9, 100.0)],
    'amstertime': [(61.4, 81.9), (59.4, 78.7)],
    'baidu': [(77.5, 87.9), (74.3, 86.6)],
}
MODELS = [
    ('full', '8192', PROJECT / 'logs/reproduction_20261028_guarded_full/lightning_logs/version_0/checkpoints/wpca8192_last.ckpt', []),
    ('192prepool', '4096', PROJECT / 'logs/reproduction_20261028_guarded_full_192/lightning_logs/version_0/checkpoints/wpca4096_last.ckpt', ['--nv_pca', '192']),
]


@lru_cache(maxsize=64)
def _hash_at_state(path, size, mtime_ns):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_hash(path):
    stat = Path(path).stat()
    return _hash_at_state(str(path), stat.st_size, stat.st_mtime_ns)


def matching_result(path, item, checkpoint):
    if not path.is_file():
        return None
    result = json.loads(path.read_text())
    if (result.get('protocol_fingerprint') != item['protocol_fingerprint']
            or result.get('manifest_sha256') != file_hash(item['manifest'])
            or result.get('checkpoint_sha256') != file_hash(checkpoint)):
        return None
    return result


def summarize(root, audit):
    rows = []
    for index, (model, _, checkpoint, _) in enumerate(MODELS):
        for item in audit:
            name = item['dataset']
            if name not in PAPER:
                continue
            path = root / model / f'{checkpoint.name}_{name}_metrics.json'
            result = matching_result(path, item, checkpoint)
            rows.append({'model': model, 'dataset': name,
                         'status': ('reused' if result.get('reused_from') else 'completed') if result else item['status'] if item['status'] == 'blocked' else 'pending',
                         'comparability': item['protocol']['comparability'],
                         'paper_R1_R5_percent': PAPER[name][index], 'paper_source': PAPER_URL,
                         'result_path': str(path) if result else None, 'result': result,
                         'audit': item['audit'], 'protocol': item['protocol']})
    (root / 'results.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    lines = ['# 本机数据适配后的真实评估结果', '',
             '复用已完成的最后一轮 + Whitening 权重。原始图像、作者名单/GT、旧 MSLS 结果保持不变。',
             '论文最终 checkpoint 的具体选择未明确；本轮延续 README 的 last checkpoint，未按测试成绩挑选。', '',
             f'论文报告值来源：[Table 1]({PAPER_URL})。AmsterTime 采用论文 61.4/59.4，README 的61.7/59.2另作来源差异。', '',
             '| 模型 | 数据集 | R@1 | R@5 | 论文 R@1 | 状态 / 可比性 |', '|---|---|---:|---:|---:|---|']
    for row in rows:
        result = row['result']
        recalls = result.get('recalls') if result else None
        r1 = f"{recalls['1'] * 100:.2f}%" if recalls else '—'
        r5 = f"{recalls['5'] * 100:.2f}%" if recalls else '—'
        lines.append(f"| {row['model']} | {row['dataset']} | {r1} | {r5} | {row['paper_R1_R5_percent'][0]:.1f}% | {row['status']} / {row['comparability']} |")
    lines.extend(['', '`author_protocol` 表示名单、真值规则和图像格式适配符合作者代码；不表示已证明历史软件逐位一致。',
                  '`conditional` 有已注明的数据来源条件，成绩单独报告；`blocked` 不计分、不填零。',
                  'SFSM 全部 1000 查询均进入 Recall 分母，包含作者 25m 规则下没有正样本的 34 条。',
                  'Baidu 使用官方 camera 原始 XYZ 的 10m 规则；全部 2292 查询进入分母，包括 44 条空正例；排除混入的 extended 数据组。',
                  'MSLS test 已开放官方元数据，可单独运行 msls_test_benchmark.sh；它不参与本九集主表，旧无 GT 清单只用于导出候选。', '',
                  '逐图 ID、实际路径、GT、协议指纹和差异见 [数据审计](dataset-audit.zh-CN.md)及 manifests；JSON 结果记录 checkpoint/manifest SHA256。'])
    (root / 'results.zh-CN.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--view-root', type=Path, default=None,
                        help='Reuse unchanged derived image views; manifests/results stay in the new output directory')
    parser.add_argument('--reuse-results-from', type=Path, default=None,
                        help='Copy only results with identical checkpoint SHA256 and exact manifest SHA256')
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if root.exists() and not args.resume:
        raise FileExistsError(f'Use a new output directory, or --resume: {root}')
    root.mkdir(parents=True, exist_ok=True)
    state = {'status': 'running', 'phase': 'prepare datasets', 'started': time.time(), 'stages': []}
    state_path = root / 'pipeline.json'

    def save():
        state_path.write_text(json.dumps(state, indent=2))

    save()
    audit = []
    try:
        audit = prepare(root, PAPER_DATASETS + ['MSLS_Test', 'pitts30k_val', 'pitts30k_test'], args.view_root)
        summarize(root, audit)
        fast_order = ['SPED', 'st_lucia', 'amstertime', 'MSLS', 'Nordland', 'sfsm', 'tokyo247', 'pitts250k_test', 'baidu']
        ready = {v['dataset'] for v in audit if v['status'] == 'ready'}
        for model, pcs, checkpoint, projection in MODELS:
            folder = root / model
            folder.mkdir(exist_ok=True)
            pending = []
            for name in fast_order:
                if name not in ready:
                    continue
                result = folder / f'{checkpoint.name}_{name}_metrics.json'
                if not result.exists() and args.reuse_results_from is not None:
                    source = args.reuse_results_from.resolve() / model / result.name
                    item = next(v for v in audit if v['dataset'] == name)
                    previous = matching_result(source, item, checkpoint)
                    if previous is not None:
                        copied = {**previous, 'reused_from': str(source),
                                  'reuse_verification': 'identical checkpoint SHA256 and manifest SHA256'}
                        result.write_text(json.dumps(copied, ensure_ascii=False, indent=2))
                        source_predictions = source.with_name(source.name.replace('_metrics.json', '_predictions.npz'))
                        if source_predictions.is_file():
                            shutil.copy2(source_predictions, folder / source_predictions.name)
                if result.is_file():
                    item = next(v for v in audit if v['dataset'] == name)
                    if matching_result(result, item, checkpoint) is None:
                        raise ValueError(f'Resume checkpoint, manifest or image inventory changed: {name}; use a new directory')
                else:
                    pending.append(name)
            if not pending:
                continue
            command = [sys.executable, '-u', 'eval.py', '--aggregation', 'NETVLAD', '--antiburst',
                       *projection, '--wpca', '--num_pcs', pcs, '--ckpt_state_dict',
                       '--resume_train', str(checkpoint), '--val_datasets', *pending,
                       '--dataset_manifest_dir', str(root / 'manifests'), '--batch_size', '32',
                       '--num_workers', '4', '--store_eval_output', '--save_dir', str(folder),
                       '--expName', f'author_protocol_{model}', '--no_wandb']
            record = {'model': model, 'command': command, 'status': 'running', 'started': time.time()}
            state['stages'].append(record)
            state['phase'] = f'evaluate {model}'
            save()
            print('Evaluating existing checkpoint:', model, pending, flush=True)
            process = subprocess.Popen(command, cwd=PROJECT)
            while process.poll() is None:
                summarize(root, audit)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            record.update(status='completed' if process.returncode == 0 else 'failed', exit_code=process.returncode, ended=time.time())
            save()
            summarize(root, audit)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        for model, _, checkpoint, _ in MODELS:
            for item in audit:
                if item['dataset'] in PAPER and item['status'] == 'ready':
                    path = root / model / f"{checkpoint.name}_{item['dataset']}_metrics.json"
                    if matching_result(path, item, checkpoint) is None:
                        raise RuntimeError(f"Incomplete ready evaluation: {model}/{item['dataset']}")
        state.update(status='completed', phase='finished', ended=time.time(),
                     blocked_datasets=[v['dataset'] for v in audit if v['status'] == 'blocked'])
    except BaseException as error:
        state.update(status='failed', error=repr(error), ended=time.time())
        raise
    finally:
        save()
        if audit:
            summarize(root, audit)


if __name__ == '__main__':
    main()
