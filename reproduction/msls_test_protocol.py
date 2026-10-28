"""Build the public MSLS test im2im/all protocol without changing raw data."""
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from scipy.spatial import cKDTree

CITIES = ['miami', 'athens', 'buenosaires', 'stockholm', 'bengaluru', 'kampala']
COMMIT = '8f950f59515bfbc69727151dcccb19e0e8a01824'
SOURCE = 'https://github.com/mapillary/mapillary_sls'
UPSTREAM_SHA256 = {
    'README.md': '39879d66ec9e611dcef281b0104e53ee89251b61275b92c68ca62529b7a3df5a',
    'evaluate.py': '0fb5ad0fc69c90f68a4b72acbaed2781974b6c9cca8f75682de9da64753087e1',
    'mapillary_sls/datasets/msls.py': '8baf7067ce21b32d9aff0fa75496cfa496dba70b9eca2dae6d5bb12c1dd4fae6',
    'mapillary_sls/datasets/generic_dataset.py': '698664cf7d825f7bf04955712899c761e6b1986c9a34e137eeaee22e41856837',
    'mapillary_sls/utils/eval.py': '5a80be2d6ef4cca56358db1136591df489349fb7ff967f96548fb54dfc0bd6bf',
    'LICENSE': 'fe415cf23cc4f7f6b664b48600a0ee2bd92415eb9332de39012c7e1433ff60c1',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def metadata_archive(repo_root):
    folder = Path(repo_root) / 'datasets/msls_test'
    candidates = [p for p in [folder / 'test_meta.zip', folder / 'msls_meta.zip'] if p.is_file()]
    if len(candidates) > 1 and len({sha256(p) for p in candidates}) != 1:
        raise ValueError('Two different MSLS test metadata archives; refusing an ambiguous source')
    return candidates[0] if candidates else None


def official_source(root):
    """Verify the retained upstream files before executing the CPU reference loader."""
    source = root / 'source'
    provenance_path = source / 'provenance.json'
    if not provenance_path.exists():
        import urllib.request
        files = []
        for name, expected in UPSTREAM_SHA256.items():
            path = source / name
            url = f'https://raw.githubusercontent.com/mapillary/mapillary_sls/{COMMIT}/{name}'
            if not path.exists():
                with urllib.request.urlopen(url, timeout=30) as response:
                    payload = response.read(1024 * 1024 + 1)
                if hashlib.sha256(payload).hexdigest() != expected:
                    raise ValueError(f'Official source download SHA256 mismatch: {name}')
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix('.download')
                temporary.write_bytes(payload)
                temporary.replace(path)
            if sha256(path) != expected:
                raise ValueError(f'Existing official source cache differs: {name}')
            files.append({'path': name, 'url': url, 'sha256': expected})
        provenance_path.write_text(json.dumps({'repository': SOURCE, 'commit': COMMIT, 'files': files}, indent=2))
    provenance = json.loads(provenance_path.read_text())
    if provenance['commit'] != COMMIT or provenance['repository'] != SOURCE:
        raise ValueError('Unexpected official MSLS code revision')
    for row in provenance['files']:
        if (row['path'] not in UPSTREAM_SHA256 or row['sha256'] != UPSTREAM_SHA256[row['path']]
                or sha256(source / row['path']) != row['sha256']):
            raise ValueError(f"Official code changed: {row['path']}")
    required = {'evaluate.py', 'mapillary_sls/datasets/msls.py',
                'mapillary_sls/datasets/generic_dataset.py', 'mapillary_sls/utils/eval.py'}
    if not required.issubset({row['path'] for row in provenance['files']}):
        raise ValueError('Incomplete official evaluator source cache')
    return source


def build_msls_test_manifest(repo_root, protocol_root=None):
    repo_root = Path(repo_root).resolve()
    root = Path(protocol_root or repo_root / 'logs/msls_test_official_protocol').resolve()
    archive = metadata_archive(repo_root)
    if archive is None:
        raise FileNotFoundError('Place official test_meta.zip in datasets/msls_test/')
    root.mkdir(parents=True, exist_ok=True)
    # Extract only verified CSV metadata into a separate view; images stay in place.
    proof, members = [], {}
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            if entry.is_dir():
                continue
            relative = Path(entry.filename)
            if (len(relative.parts) != 4 or relative.parts[0] != 'test'
                    or relative.parts[1] not in CITIES
                    or relative.parts[2] not in ['query', 'database']
                    or relative.name not in ['raw.csv', 'postprocessed.csv', 'seq_info.csv',
                                             'subtask_index.csv', 'extra_data.csv']):
                raise ValueError(f'Unexpected metadata member: {entry.filename}')
            if entry.filename in members:
                raise ValueError('Duplicate metadata ZIP member')
            payload = bundle.read(entry)  # zipfile checks the original CRC.
            members[entry.filename] = payload
            target = root / 'data' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != payload:
                raise ValueError(f'Refusing to replace changed metadata view: {target}')
            if not target.exists():
                target.write_bytes(payload)
            proof.append({'id': entry.filename, 'size': len(payload),
                          'crc32': f'{entry.CRC:08x}',
                          'sha256': hashlib.sha256(payload).hexdigest()})
    if len(proof) != 60:
        raise ValueError(f'Expected complete six-city metadata, found {len(proof)} CSV files')
    provenance = {'archive': str(archive), 'archive_sha256': sha256(archive),
                  'verified_csv_count': len(proof), 'files': proof}
    (root / 'metadata-provenance.json').write_text(json.dumps(provenance, indent=2))

    def selected(city, role):
        prefix = f'test/{city}/{role}/'
        tables = {name: list(csv.DictReader(io.StringIO(members[prefix + name].decode('utf-8'))))
                  for name in ['postprocessed.csv', 'raw.csv', 'subtask_index.csv', 'seq_info.csv']}
        post = tables['postprocessed.csv']
        for name, rows in tables.items():
            if len(rows) != len(post) or any(
                    row['key'] != base['key'] or (name != 'seq_info.csv' and int(row['']) != i)
                    for i, (row, base) in enumerate(zip(rows, post))):
                raise ValueError(f'Misaligned MSLS tables: {city}/{role}/{name}')
        if len({row['key'] for row in post}) != len(post):
            raise ValueError('Duplicate image keys in metadata')
        ids, coords, excluded = [], [], []
        for index, row in enumerate(post):
            task = tables['subtask_index.csv'][index]['all']
            pano = tables['raw.csv'][index]['pano']
            if task not in ['True', 'False'] or pano not in ['True', 'False']:
                raise ValueError('Unexpected boolean metadata')
            identifier = prefix + 'images/' + row['key'] + '.jpg'
            if task == 'True' and pano == 'False':
                ids.append(identifier)
                coords.append([float(row['easting']), float(row['northing'])])
            elif task == 'True':
                excluded.append(identifier)
        xy = np.asarray(coords, dtype=np.float64)
        if xy.shape != (len(ids), 2) or not np.isfinite(xy).all():
            raise ValueError('Invalid UTM coordinates')
        return ids, xy, excluded

    refs, raw_queries, positives, qidx, no_match, city_audit, panos = [], [], [], [], [], [], []
    for city in CITIES:
        db, db_xy, db_panos = selected(city, 'database')
        query, query_xy, _ = selected(city, 'query')
        offset, qoffset = len(refs), len(raw_queries)
        local_positives = cKDTree(db_xy).query_ball_point(query_xy, r=25.0)
        refs.extend(db)
        raw_queries.extend(query)
        panos.extend(db_panos)
        for i, row in enumerate(local_positives):
            if row:
                qidx.append(qoffset + i)
                positives.append([offset + int(j) for j in sorted(row)])
            else:
                no_match.append(query[i])
        city_audit.append({'city': city, 'references': len(db), 'raw_queries': len(query),
                           'scored_queries': sum(bool(row) for row in local_positives),
                           'queries_without_25m_match': sum(not row for row in local_positives)})

    # Execute upstream selection and GT independently, with only a NumPy ragged-list
    # compatibility edit. No training method or image __getitem__ is called.
    source = official_source(root)
    original = source / 'mapillary_sls/datasets/msls.py'
    compatible = original.read_text()
    edits = {}
    for field in ['pIdx', 'nonNegIdx']:
        before = f'self.{field} = np.asarray(self.{field})'
        after = f'self.{field} = np.asarray(self.{field}, dtype=object)'
        if compatible.count(before) != 1:
            raise ValueError('Unexpected upstream compatibility edit target')
        compatible = compatible.replace(before, after)
        edits[before] = after
    target = original.with_name('msls_numpy_compatible.py')
    target.write_text(compatible)
    sys.path.insert(0, str(source))
    try:
        spec = importlib.util.spec_from_file_location('official_msls_numpy_compatible', target)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        upstream = module.MSLS(root / 'data', cities='test', mode='val', task='im2im',
                               seq_length=1, subtask='all', posDistThr=25)
    finally:
        sys.path.remove(str(source))
    normalize = lambda path: 'test/' + str(path).split('/test/', 1)[1]
    checks = {
        'reference_ids_and_order': list(map(normalize, upstream.dbImages)) == refs,
        'raw_query_ids_and_order': list(map(normalize, upstream.qImages)) == raw_queries,
        'query_indices': list(map(int, upstream.qIdx)) == qidx,
        'all_positive_sets': [list(map(int, row)) for row in upstream.pIdx] == positives,
        'excluded_no_match_queries': list(upstream.query_keys_with_no_match) == [Path(i).stem for i in no_match],
    }
    if not all(checks.values()):
        raise ValueError(f'Independent implementation differs from official MSLS loader: {checks}')
    legacy = repo_root / 'datasets/msls_test'
    old_refs = list(map(str, np.load(legacy / 'msls_test_dbImages.npy', allow_pickle=True)))
    old_queries = list(map(str, np.load(legacy / 'msls_test_qImages.npy', allow_pickle=True)))
    reference_set = set(refs)
    excluded_legacy_refs = [identifier for identifier in old_refs if identifier not in reference_set]
    excluded_legacy_set = set(excluded_legacy_refs)
    if ([identifier for identifier in old_refs if identifier not in excluded_legacy_set] != refs
            or old_queries != raw_queries or not set(excluded_legacy_refs).issubset(panos)):
        raise ValueError('Legacy image lists disagree beyond the explicitly identified panorama exclusions')
    queries = [raw_queries[i] for i in qidx]
    rows = lambda ids: [{'id': identifier, 'path': str(repo_root / 'datasets/msls_test' / identifier[5:])}
                        for identifier in ids]
    audit = {'expected_references': len(refs), 'expected_queries': len(queries),
             'raw_queries': len(raw_queries), 'official_query_indices': qidx,
             'legacy_reference_count': len(old_refs), 'legacy_query_count': len(old_queries),
             'excluded_legacy_panorama_reference_ids': excluded_legacy_refs,
             'excluded_queries_without_25m_positive': no_match,
             'per_city': city_audit, 'official_loader_verification': checks,
             'source_code_commit': COMMIT, 'numpy_compatibility_edits': edits,
             'metadata_archive': str(archive), 'metadata_archive_sha256': provenance['archive_sha256'],
             'source_images_modified': False, 'verified_metadata_csv_count': len(proof),
             'metadata_file_proofs': proof,
             'legacy_npy_sha256': {name: sha256(legacy / name) for name in
                                   ['msls_test_dbImages.npy', 'msls_test_qImages.npy']}}
    (root / 'verification.json').write_text(json.dumps(audit, indent=2))
    return {'schema_version': 1, 'dataset': 'MSLS_Test', 'status': 'ready',
            'references': rows(refs), 'queries': rows(queries), 'positives': positives,
            'protocol': {
                'name': 'MSLS official public test, im2im/all, 25m',
                'source': f'{SOURCE}/tree/{COMMIT}', 'comparability': 'official_test_protocol',
                'positive_definition': 'Same-city Euclidean distance between original UTM coordinates <= 25m',
                'task': 'im2im', 'subtask': 'all', 'cities': CITIES, 'exclude_panos': True,
                'notes': ['Official mode=val with cities=test uses released UTM ground truth, not the no-GT mode=test.',
                          'Use subtask_index all=True and raw pano=False; exclude queries with no positive exactly as official qIdx.',
                          'Independent selection and every positive set match the pinned official MSLS loader.',
                          'Legacy author candidate lists have 7 extra Kampala panorama references and 32 no-positive queries; originals are retained.',
                          'This is an additional MSLS test result, separate from the paper MSLS validation table.']},
            'audit': audit}
