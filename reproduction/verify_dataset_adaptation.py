"""Small CPU regressions for manifests, empty positives, unavailable GT and streaming."""
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset
import eval as evaluator
from dataloaders.val.ManifestDataset import ManifestDataset
from reproduction.prepare_datasets import finalize_manifest
from reproduction.benchmark import matching_result, file_hash
from utils.validation import get_validation_recalls


def main():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        entries = []
        for i in range(5):
            path = root / f'{i}.png'
            Image.new('RGBA', (8, 9), (i, 0, 0, 255)).save(path)
            entries.append({'id': f'author-{i}', 'path': str(path)})
        manifest = finalize_manifest({'dataset': 'fixture', 'status': 'ready',
            'references': entries[:2], 'queries': entries[2:], 'positives': [[0], [], [1]],
            'protocol': {'name': 'fixture', 'comparability': 'author_protocol'}, 'audit': {}})
        path = root / 'fixture.json'
        path.write_text(json.dumps(manifest))
        dataset = ManifestDataset(path)
        assert dataset.reference_ids == ['author-0', 'author-1']
        assert dataset[0][0].mode == 'RGB'
        assert len(dataset.ground_truth) == 3 and len(dataset.ground_truth[1]) == 0
        assert manifest['audit']['queries_without_positives'] == 1
        refs = np.eye(2, dtype='float32')
        queries = refs[[0, 0, 1]]
        _, _, recalls = get_validation_recalls(refs, queries, [1], dataset.ground_truth,
                                                print_results=False, retrieval_batch_size=1)
        assert recalls[1] == 2 / 3  # Do not silently drop the empty-positive query.
        _, predictions, unavailable = get_validation_recalls(refs, queries, [1], None,
                                                               print_results=False, retrieval_batch_size=1)
        assert unavailable is None and predictions.shape == (3, 1)
        x = torch.arange(5, dtype=torch.float32).reshape(-1, 1)
        loader = DataLoader(TensorDataset(x, torch.arange(5)), batch_size=2)
        model = lambda values: torch.cat([values, values + 10], dim=1)
        evaluator.args = SimpleNamespace(device='cpu', useFC=True)
        with patch.object(evaluator.torch, 'autocast', lambda **kw: contextlib.nullcontext()):
            mapped = evaluator.get_descriptors(model, loader, root / 'features.dat')
        np.testing.assert_array_equal(mapped, model(x).numpy())
        mapped._mmap.close()
        checkpoint = root / 'checkpoint.bin'
        checkpoint.write_bytes(b'checkpoint A')
        item = {'manifest': str(path), 'protocol_fingerprint': manifest['protocol_fingerprint']}
        result = root / 'metrics.json'
        result.write_text(json.dumps({'protocol_fingerprint': manifest['protocol_fingerprint'],
            'manifest_sha256': file_hash(path), 'checkpoint_sha256': file_hash(checkpoint)}))
        assert matching_result(result, item, checkpoint) is not None
        path.write_text(json.dumps({**manifest, 'audit': {'changed': True}}))
        assert matching_result(result, item, checkpoint) is None
        path.write_text(json.dumps(manifest))
        checkpoint.write_bytes(b'checkpoint B changed')
        assert matching_result(result, item, checkpoint) is None
        manifest['status'] = 'blocked'
        path.write_text(json.dumps(manifest))
        try:
            ManifestDataset(path)
        except ValueError:
            pass
        else:
            raise AssertionError('Blocked protocol must not be evaluated')
    print('PASS: author order, RGB, full query denominator, missing GT, bounded feature storage and stale-result rejection')


if __name__ == '__main__':
    main()
