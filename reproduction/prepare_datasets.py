"""Build audited evaluation manifests and derived views without modifying raw data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
PAPER_DATASETS = ['MSLS', 'Nordland', 'pitts250k_test', 'SPED', 'sfsm',
                  'tokyo247', 'st_lucia', 'amstertime', 'baidu']


def finalize_manifest(manifest):
    refs, queries = manifest.get('references', []), manifest.get('queries', [])
    audit = manifest.setdefault('audit', {})
    if manifest['status'] == 'ready':
        for label, entries in [('references', refs), ('queries', queries)]:
            missing = [v['path'] for v in entries if not Path(v['path']).is_file()]
            audit[f'missing_{label}'] = len(missing)
            expected = audit.get(f'expected_{label}', len(entries))
            if missing or len(entries) != expected or len({v['id'] for v in entries}) != len(entries):
                raise ValueError(f"{manifest['dataset']}: invalid {label}: missing={missing[:3]}, count={len(entries)}, expected={expected}")
        if not refs or not queries:
            raise ValueError('Empty evaluation split')
        positives = manifest['positives']
        if positives is not None:
            if len(positives) != len(queries) or any(j < 0 or j >= len(refs) for row in positives for j in row):
                raise ValueError(f"{manifest['dataset']}: invalid ground truth")
            audit['queries_without_positives'] = sum(not row for row in positives)
            audit['positive_edges'] = sum(len(row) for row in positives)
            audit['recall_denominator'] = len(queries)
        else:
            audit['recall_denominator'] = None
        sampled = []
        inventory = hashlib.sha256()
        for entry in refs + queries:
            path = Path(entry['path'])
            stat = path.stat()
            inventory.update(f"{entry['id']}\0{path}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        audit['image_inventory_sha256'] = inventory.hexdigest()
        audit['image_inventory_definition'] = 'ordered image ID, resolved path, file size, mtime_ns; not a pixel-content hash'
        for entries in [refs, queries]:
            for index in sorted({0, len(entries) // 2, len(entries) - 1}):
                with Image.open(entries[index]['path']) as image:
                    image.convert('RGB').load()
                sampled.append(entries[index]['id'])
        audit['decoded_sample_ids'] = sampled
        audit['decode_check'] = 'first/middle/last per split; all images decoded during evaluation'
    canonical = {'references': [v['id'] for v in refs], 'queries': [v['id'] for v in queries],
                 'positives': manifest.get('positives'), 'protocol': manifest['protocol']}
    manifest['protocol_fingerprint'] = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    manifest['schema_version'] = 1
    return manifest


def prepare(output_dir, names=None, view_root=None):
    from reproduction.adapt_native import build_native
    from reproduction.adapt_retrieval import build_retrieval, materialize_retrieval_views
    output_dir = Path(output_dir).resolve()
    manifests = output_dir / 'manifests'
    views = Path(view_root).resolve() if view_root is not None else output_dir / 'views'
    manifests.mkdir(parents=True, exist_ok=True)
    views.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in names or PAPER_DATASETS:
        print(f'Preparing {name}: preserve author image IDs and ground truth', flush=True)
        manifest = build_native(name, PROJECT, views)
        if manifest is None:
            manifest = build_retrieval(name, PROJECT, views)
        if manifest is None:
            raise ValueError(f'Unknown dataset {name}')
        if manifest['status'] == 'ready':
            materialize_retrieval_views(manifest, workers=4)
        manifest = finalize_manifest(manifest)
        target = manifests / f'{name.lower()}.json'
        target.write_text(json.dumps(manifest, ensure_ascii=False, separators=(',', ':')))
        summary.append({'dataset': name, 'manifest': str(target), 'status': manifest['status'],
                        'protocol': manifest['protocol'], 'audit': manifest['audit'],
                        'references': len(manifest.get('references', [])),
                        'queries': len(manifest.get('queries', [])),
                        'protocol_fingerprint': manifest['protocol_fingerprint']})
        print(f"{name}: {manifest['status']}, {len(manifest.get('references', []))} refs / {len(manifest.get('queries', []))} queries", flush=True)
    (output_dir / 'dataset-audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    lines = ['# 本机数据与作者协议核对', '', '原始图像和作者 NPY/GT 保持不变。派生视图与名单位于本目录。', '',
             '| 数据集 | 参考图 | 查询图 | 状态 | 无正样本查询 |', '|---|---:|---:|---|---:|']
    for item in summary:
        a = item['audit']
        lines.append(f"| {item['dataset']} | {item['references']} | {item['queries']} | {item['status']} / {item['protocol']['comparability']} | {a.get('queries_without_positives', '—')} |")
    for item in summary:
        lines.extend(['', f"## {item['dataset']}", '', *[f'- {note}' for note in item['protocol'].get('notes', [])]])
    (output_dir / 'dataset-audit.zh-CN.md').write_text('\n'.join(lines) + '\n')
    # Canonical training-loader view: the source release now lives in the repo.
    msls_view = views / 'MSLS'
    msls_view.mkdir(exist_ok=True)
    for role, source in [('train_val', PROJECT / 'datasets/msls_val'), ('test', PROJECT / 'datasets/msls_test')]:
        target = msls_view / role
        if not target.exists():
            target.symlink_to(source, target_is_directory=True)
    for alias, destination in [('active_dataset_manifests', manifests), ('active_dataset_views', views)]:
        link = PROJECT / 'logs' / alias
        if link.exists() and not link.is_symlink():
            raise FileExistsError(f'Refusing to replace existing directory {link}')
        temporary = link.with_name(link.name + '.tmp')
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(destination, target_is_directory=True)
        temporary.replace(link)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', default=PAPER_DATASETS)
    args = parser.parse_args()
    prepare(args.output_dir, args.datasets)
