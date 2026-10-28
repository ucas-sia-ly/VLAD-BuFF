"""Evaluate both existing checkpoints on the now-public official MSLS test GT."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from reproduction.benchmark import MODELS, file_hash, matching_result
from reproduction.msls_test_protocol import build_msls_test_manifest, official_source
from reproduction.prepare_datasets import finalize_manifest
from reproduction.export_trainroom_snapshot import export_snapshot


def verify_prediction(folder, checkpoint, manifest, result):
    path = folder / f'{checkpoint.name}_MSLS_Test_predictions.npz'
    with np.load(path) as archive:
        predictions = archive['predictions']
        distances = archive['distances']
        refs, queries = [r['id'] for r in manifest['references']], [r['id'] for r in manifest['queries']]
        if (list(archive['reference_ids']) != refs or list(archive['query_ids']) != queries
                or predictions.shape != (len(queries), 10) or distances.shape != predictions.shape
                or np.any(predictions < 0) or np.any(predictions >= len(refs))
                or not np.isfinite(distances).all() or np.any(np.diff(distances, axis=1) < 0)
                or any(len(set(row)) != len(row) for row in predictions)):
            raise ValueError('Prediction inventory, shape or ranking is invalid')
        recomputed = {str(k): sum(bool(set(row[:k]) & set(gt)) for row, gt in
                                      zip(predictions, manifest['positives'])) / len(queries)
                      for k in [1, 5, 10]}
        source = official_source(PROJECT / 'logs/msls_test_official_protocol')
        spec = importlib.util.spec_from_file_location('official_msls_eval', source / 'mapillary_sls/utils/eval.py')
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        db_keys = np.asarray([Path(identifier).stem for identifier in refs])
        q_keys = np.asarray([Path(identifier).stem for identifier in queries])
        if len(set(db_keys)) != len(db_keys) or len(set(q_keys)) != len(q_keys):
            raise ValueError('Official prediction format requires globally unique image keys')
        text_predictions = np.concatenate([q_keys[:, None], db_keys[predictions]], axis=1)
        official = upstream.eval(q_keys, [[db_keys[i] for i in row] for row in manifest['positives']],
                                 text_predictions, ks=[1, 5, 10])
        for k in [1, 5, 10]:
            if not (np.isclose(recomputed[str(k)], result['recalls'][str(k)], rtol=0, atol=1e-12)
                    and np.isclose(official[f'recall@{k}'], recomputed[str(k)], rtol=0, atol=1e-12)):
                raise ValueError('Recorded, recomputed and official scorer Recall differ')
        np.savetxt(folder / 'MSLS_Test_official_top10.txt', text_predictions, fmt='%s')
    proof = {'predictions_sha256': file_hash(path), 'queries': len(queries), 'references': len(refs),
             'recorded_recall_matches_predictions': True, 'official_scorer_matches': True,
             'official_metrics': {k: float(v) for k, v in official.items()},
             'rankings_and_image_ids_valid': True,
             'checkpoint_sha256': result['checkpoint_sha256'],
             'manifest_sha256': result['manifest_sha256']}
    (folder / 'verification.json').write_text(json.dumps(proof, indent=2))
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if root.exists() and not args.resume:
        raise FileExistsError(f'Use a new output directory or --resume: {root}')
    root.mkdir(parents=True, exist_ok=True)
    state = {'status': 'running', 'phase': 'prepare official test protocol', 'started': time.time(), 'stages': []}
    rows = []

    def save():
        target = root / 'pipeline.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(target)

    save()
    try:
        manifest = finalize_manifest(build_msls_test_manifest(PROJECT))
        manifests = root / 'manifests'
        manifests.mkdir(exist_ok=True)
        target = manifests / 'msls_test.json'
        text = json.dumps(manifest, ensure_ascii=False, separators=(',', ':'))
        if target.exists() and target.read_text() != text:
            raise ValueError('Manifest changed; use a new output directory')
        target.write_text(text)
        (root / 'dataset-audit.json').write_text(json.dumps(manifest['audit'], indent=2))
        item = {'manifest': str(target), 'protocol_fingerprint': manifest['protocol_fingerprint']}
        print(f"Official test protocol: {len(manifest['references'])} references, {len(manifest['queries'])} scored queries", flush=True)
        for model, pcs, checkpoint, projection in MODELS:
            folder = root / model
            folder.mkdir(exist_ok=True)
            metrics_path = folder / f'{checkpoint.name}_MSLS_Test_metrics.json'
            result = matching_result(metrics_path, item, checkpoint)
            if metrics_path.exists() and result is None:
                raise ValueError('Existing metrics have different source provenance')
            if result is None:
                command = [sys.executable, '-u', 'eval.py', '--aggregation', 'NETVLAD', '--antiburst',
                           *projection, '--wpca', '--num_pcs', pcs, '--ckpt_state_dict',
                           '--resume_train', str(checkpoint), '--val_datasets', 'MSLS_Test',
                           '--dataset_manifest_dir', str(manifests), '--batch_size', '32',
                           '--num_workers', '4', '--store_eval_output', '--save_dir', str(folder),
                           '--expName', f'official_msls_test_{model}', '--no_wandb']
                stage = {'model': model, 'command': command, 'started': time.time(), 'status': 'running'}
                state['stages'].append(stage)
                state['phase'] = f'evaluate {model}'
                save()
                print('Evaluating existing checkpoint:', model, flush=True)
                process = subprocess.run(command, cwd=PROJECT)
                stage.update(status='completed' if process.returncode == 0 else 'failed',
                             exit_code=process.returncode, ended=time.time())
                save()
                if process.returncode:
                    raise subprocess.CalledProcessError(process.returncode, command)
                result = matching_result(metrics_path, item, checkpoint)
                if result is None:
                    raise ValueError('New metrics are missing or provenance does not match')
            proof = verify_prediction(folder, checkpoint, manifest, result)
            rows.append({'model': model, 'dataset': 'MSLS_Test', 'status': 'completed',
                         'comparability': manifest['protocol']['comparability'],
                         'result_path': str(metrics_path), 'result': result, 'verification': proof})
            (root / 'results.json').write_text(json.dumps(rows, indent=2))
            export_snapshot(root / 'results.json', root / 'trainroom')
        lines = ['# MSLS test 官方公开真值复评', '',
                 '使用同一批已训练并 Whitening 的 last 权重，不重新训练或按 test 分数选择模型。',
                 '这是额外的 MSLS test 结果；论文主表使用 MSLS val，二者单独报告。', '',
                 '| 模型 | R@1 | R@5 | R@10 |', '|---|---:|---:|---:|']
        for row in rows:
            rec = row['result']['recalls']
            lines.append(f"| {row['model']} | {rec['1'] * 100:.2f}% | {rec['5'] * 100:.2f}% | {rec['10'] * 100:.2f}% |")
        lines.extend(['', '官方协议：六城市、im2im、all、25m、排除 panorama；36,558 参考图，1,103 条计分查询。',
                      '旧作者名单为 36,565 / 1,135；按官方规则排除 7 张 Kampala panorama 和 32 条没有 25m 正例的查询。',
                      '32 条查询从官方 qIdx 排除是 MSLS 官方规则；其他九集沿用各自已有协议，没有同步改动分母。',
                      '全部图像顺序、全部正例集合与固定版本官方 loader 一致；两模型预测重算 Recall 与官方 scorer 完全一致。',
                      '推理为 322×322、FP16 autocast、单图描述子，CPU float32 IndexFlatL2 精确检索。',
                      '', '[数据审计](dataset-audit.json) · [完整指标和核验](results.json)',
                      '[官方说明](https://github.com/mapillary/mapillary_sls#-evaluate-on-the-test-set)', ''])
        (root / 'results.zh-CN.md').write_text('\n'.join(lines))
        # Publish only this completed manifest while preserving the active nine-set suite.
        # Clone the small JSON files into a new directory, never edit historical manifests.
        active = PROJECT / 'logs/active_dataset_manifests'
        combined = root / 'combined_manifests'
        combined.mkdir(exist_ok=True)
        if active.is_dir():
            for path in active.glob('*.json'):
                if path.name != 'msls_test.json':
                    destination = combined / path.name
                    if destination.exists() and destination.read_bytes() != path.read_bytes():
                        raise ValueError('Combined manifest source changed')
                    if path.resolve() != destination.resolve():
                        shutil.copy2(path, destination)
        shutil.copy2(target, combined / target.name)
        if active.exists() and not active.is_symlink():
            raise ValueError('Refusing to replace a non-symlink active directory')
        temporary = active.with_name(active.name + '.tmp')
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(combined, target_is_directory=True)
        temporary.replace(active)
        state.update(status='completed', phase='finished', ended=time.time())
    except BaseException as exc:
        state.update(status='failed', error=repr(exc), ended=time.time())
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
