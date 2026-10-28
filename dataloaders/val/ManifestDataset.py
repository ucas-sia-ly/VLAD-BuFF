"""Evaluation on audited author-ordered image lists and explicit ground truth."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    def __init__(self, manifest_path, input_transform=None):
        self.manifest_path = Path(manifest_path).resolve()
        raw = self.manifest_path.read_bytes()
        manifest = json.loads(raw)
        if manifest.get('status') != 'ready':
            raise ValueError(f"Dataset protocol is blocked: {manifest.get('dataset')}: {manifest.get('protocol')}")
        self.protocol = manifest['protocol']
        self.audit = manifest['audit']
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self.protocol_fingerprint = manifest['protocol_fingerprint']
        self.reference_ids = [v['id'] for v in manifest['references']]
        self.query_ids = [v['id'] for v in manifest['queries']]
        self.images = [v['path'] for v in manifest['references'] + manifest['queries']]
        self.num_references = len(self.reference_ids)
        self.num_queries = len(self.query_ids)
        self.ground_truth = None if manifest['positives'] is None else [np.asarray(v, dtype=np.int64) for v in manifest['positives']]
        if self.ground_truth is not None:
            if len(self.ground_truth) != self.num_queries:
                raise ValueError('Ground truth/query count mismatch')
            if any(np.any((v < 0) | (v >= self.num_references)) for v in self.ground_truth):
                raise ValueError('Positive index outside reference list')
        self.input_transform = input_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        with Image.open(self.images[index]) as source:
            image = source.convert('RGB')
        if self.input_transform is not None:
            image = self.input_transform(image)
        return image, index

    def save_predictions(self, predictions, path):
        with open(path, 'w') as stream:
            for query, matches in zip(self.query_ids, predictions):
                reference_ids = ' '.join(Path(self.reference_ids[int(i)]).stem for i in matches)
                stream.write(f'{Path(query).stem} {reference_ids}\n')
