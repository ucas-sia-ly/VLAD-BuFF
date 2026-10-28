"""Use the original CRC-verified VPR-Bench release without image transcoding."""
import binascii
import hashlib
import json
from pathlib import Path

import numpy as np


def build_nordland_official_manifest(repo_root, metadata_root):
    repo_root, metadata_root = Path(repo_root).resolve(), Path(metadata_root).resolve()
    provenance_path = metadata_root / 'provenance.json'
    provenance = json.loads(provenance_path.read_text())
    if (provenance.get('status') != 'complete'
            or provenance.get('verified_image_crc_count') != 30352
            or provenance.get('ground_truth_matches_author') is not True
            or provenance.get('image_transcoding') is not False):
        raise ValueError('Official Nordland release is not fully verified')
    root = Path(provenance['extraction_root']).resolve()
    original = repo_root / 'datasets/Nordland'
    arrays = {}
    for filename in ['Nordland_dbImages.npy', 'Nordland_qImages.npy', 'Nordland_gt.npy']:
        path = original / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != provenance['author_metadata_sha256'][filename]:
            raise ValueError(f'Author metadata changed: {filename}')
        arrays[filename] = np.load(path, allow_pickle=True)
    refs, queries = arrays['Nordland_dbImages.npy'], arrays['Nordland_qImages.npy']
    positives = [list(map(int, row)) for row in arrays['Nordland_gt.npy']]
    gt_path = Path(provenance['ground_truth_path'])
    if hashlib.sha256(gt_path.read_bytes()).hexdigest() != provenance['official_ground_truth_sha256']:
        raise ValueError('Official ground-truth file changed')
    official_gt = np.load(gt_path, allow_pickle=True, encoding='latin1')
    if len(official_gt) != len(positives) or any(int(row[0]) != i or not np.array_equal(row[1], positives[i])
                                               for i, row in enumerate(official_gt)):
        raise ValueError('Official and author Nordland ground truth differ')
    inventory_path = metadata_root / 'verified-images.jsonl'
    if hashlib.sha256(inventory_path.read_bytes()).hexdigest() != provenance['verified_inventory_sha256']:
        raise ValueError('Official image verification inventory changed')
    inventory = {row['id']: row for row in map(json.loads, inventory_path.read_text().splitlines())}
    if set(inventory) != set(map(str, np.concatenate((refs, queries)))):
        raise ValueError('Official image inventory differs from the author image IDs')
    records = {}
    for role, ids in [('references', refs), ('queries', queries)]:
        values = []
        for value in ids:
            identifier = str(value)
            image_path = (root / identifier).resolve()
            if not image_path.is_relative_to(root):
                raise ValueError('Image path escapes the official view')
            proof = inventory[identifier]
            payload = image_path.read_bytes()
            if (len(payload) != proof['size']
                    or hashlib.sha256(payload).hexdigest() != proof['sha256']
                    or (binascii.crc32(payload) & 0xFFFFFFFF) != int(proof['zip_crc32'], 16)):
                raise ValueError(f'Official image content changed: {identifier}')
            values.append({'id': identifier, 'path': str(image_path)})
        records[role] = values
    return {'schema_version': 1, 'dataset': 'Nordland', 'status': 'ready',
            **records, 'positives': positives,
            'protocol': {
                'name': 'VLAD-BuFF author Nordland protocol on original VPR-Bench release',
                'source': provenance['source_url'],
                'positive_definition': 'Original Nordland_gt.npy and release GT agree: query i -> ref 10*i +/- 1, clipped at boundaries',
                'comparability': 'author_protocol',
                'source_roles': {'references': 'winter, official ref/', 'queries': 'summer, official query/'},
                'notes': ['Use the original official 27592 reference / 2760 query image bytes with no transcoding.',
                          'Official image SHA256 and ZIP CRC verified for every selected image.',
                          'The separately archived v1 local JPEG75 scores are not substituted for these release scores.'],
                'archive_sha256': provenance['archive_sha256']},
            'audit': {'expected_references': 27592, 'expected_queries': 2760,
                      'missing_references': 0, 'missing_queries': 0,
                      'source_provenance': str(provenance_path),
                      'source_provenance_sha256': hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
                      'verified_image_crc_and_sha256_count': len(inventory),
                      'official_ground_truth_matched_queries': len(official_gt),
                      'image_transcoding': False}}
