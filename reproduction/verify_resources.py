"""Regression checks and optional full-shape validation storage test. No training."""
import argparse
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import pytorch_lightning as pl
from add_pca import pca
from utils.bounded_pca import disk_pca, disk_whiten
from utils.validation import get_validation_recalls
from vpr_model import VPRModel


def regression_checks():
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as folder:
        for shape in [(30, 12), (12, 30)]:
            x = rng.normal(size=shape)
            components, values, mean = pca(x.copy(), 5)
            data = np.memmap(Path(folder) / 'features', mode='w+', dtype='float64', shape=shape)
            data[:] = x
            actual, actual_values, actual_mean = disk_pca(data, 5, block_size=4)
            np.testing.assert_allclose(actual_values, values, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(actual_mean, mean, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(actual @ actual.T, components @ components.T, rtol=1e-8, atol=1e-8)
            whitened = disk_whiten(actual, actual_values, str(Path(folder) / 'whitened'), block_size=3)
            np.testing.assert_allclose(whitened, actual * (1 / np.sqrt(actual_values + 1e-9)), rtol=1e-12, atol=1e-12)
            del data, actual, whitened
        refs = rng.normal(size=(200, 32)).astype('float32')
        queries = rng.normal(size=(50, 32)).astype('float32')
        gt = [np.array([i]) for i in range(len(queries))]
        for gpu in [False, True]:
            a = get_validation_recalls(refs, queries, [1, 5, 10], gt, print_results=False, faiss_gpu=gpu)
            b = get_validation_recalls(refs, queries, [1, 5, 10], gt, print_results=False, faiss_gpu=gpu,
                                       retrieval_batch_size=13)
            np.testing.assert_array_equal(a[1], b[1])
            np.testing.assert_allclose(a[0], b[0], rtol=1e-3, atol=1e-3)
            assert a[2] == b[2]
    print('PASS: float64 PCA subspace/whitening and CPU/GPU chunked retrieval regression checks', flush=True)


def capacity_check(full_size):
    sizes = [(17608, 10000), (16816, 10000), (19611, 18871)] if full_size else [(60, 40)]
    dim = 49152 if full_size else 32
    class Dataset:
        def __init__(self, n, nr):
            self.n = n
            self.num_references = nr
            self.pIdx = [np.array([0])] * (n - nr)
        def __len__(self):
            return self.n
    datasets = [Dataset(n, nr) for n, nr in sizes]
    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'logs') as folder:
        model = VPRModel.__new__(VPRModel)
        pl.LightningModule.__init__(model)
        model._trainer = SimpleNamespace(log_dir=folder, datamodule=SimpleNamespace(
            val_datasets=datasets, val_set_names=[f'msls_synthetic_capacity_{i}' for i in range(len(sizes))]))
        model.args = SimpleNamespace(no_wandb=True)
        model.faiss_gpu = True
        model.forward = lambda images: torch.zeros(len(images), dim)
        results = {}
        model.log = lambda name, value, **kw: results.update({name: value})
        model.on_validation_epoch_start()
        for i, (n, _) in enumerate(sizes):
            for start in range(0, n, 60):
                batch = (torch.zeros(min(60, n - start), 1), None)
                if len(sizes) == 1:
                    model.validation_step(batch, start // 60)  # single-loader default index
                else:
                    model.validation_step(batch, start // 60, i)
            assert model.val_offsets[i] == n
        with contextlib.redirect_stdout(io.StringIO()):
            model.on_validation_epoch_end()
        assert len(results) == 3 * len(sizes)
        assert not list(Path(folder).rglob('*.dat'))
    print(f'PASS: validation storage + GPU retrieval + cleanup: {sizes}, dimension={dim}; synthetic data only', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-size', action='store_true')
    args = parser.parse_args()
    regression_checks()
    capacity_check(False)
    if args.full_size:
        capacity_check(True)
