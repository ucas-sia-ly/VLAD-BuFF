import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import argparse
import wandb
import random
import numpy as np
from collections import OrderedDict
from vpr_model import VPRModel
from utils.validation import get_validation_recalls
import os
import json
import hashlib
from pathlib import Path

VAL_DATASETS = [
    "MSLS",
    "MSLS_Test",
    "pitts30k_test",
    "pitts250k_test",
    "Nordland",
    "SPED",
    "pitts30k_val",
    "st_lucia",
    "tokyo247",
    "amstertime",
    "baidu",
    "sfsm",
]


def input_transform(image_size=None):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    if image_size:
        return T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD),
            ]
        )
    else:
        return T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])


def get_val_dataset(dataset_name, image_size=None, manifest_dir=None):
    manifest_dir = manifest_dir or os.environ.get("VLADBUFF_MANIFEST_DIR")
    default_manifests = Path(__file__).resolve().parent / "logs/active_dataset_manifests"
    if manifest_dir is None and default_manifests.is_dir():
        manifest_dir = default_manifests
    if manifest_dir:
        from dataloaders.val.ManifestDataset import ManifestDataset
        manifest_path = Path(manifest_dir) / f"{dataset_name.lower()}.json"
        ds = ManifestDataset(manifest_path, input_transform(image_size))
        return ds, ds.num_references, ds.num_queries, ds.ground_truth
    dataset_name = dataset_name.lower()
    transform = input_transform(image_size=image_size)

    if "nordland" in dataset_name:
        from dataloaders.val.NordlandDataset import NordlandDataset
        ds = NordlandDataset(input_transform=transform)

    elif "msls_test" in dataset_name:
        from dataloaders.val.MapillaryTestDataset import MSLSTest
        ds = MSLSTest(input_transform=transform)

    elif "msls" in dataset_name:
        from dataloaders.val.MapillaryDataset import MSLS
        ds = MSLS(input_transform=transform)

    elif "pitts" in dataset_name:
        from dataloaders.val.PittsburghDataset import PittsburghDataset
        ds = PittsburghDataset(which_ds=dataset_name, input_transform=transform)

    elif "sped" in dataset_name:
        from dataloaders.val.SPEDDataset import SPEDDataset
        ds = SPEDDataset(input_transform=transform)

    elif "st_lucia" in dataset_name:
        from dataloaders.val.StluciaDataset import StluciaDataset
        ds = StluciaDataset(input_transform=transform)

    elif "tokyo247" in dataset_name:
        from dataloaders.val.Tokyo247Dataset import Tokyo247Dataset
        ds = Tokyo247Dataset(input_transform=transform)

    elif "sfsm" in dataset_name:
        from dataloaders.val.SfsmDataset import SfsmDataset
        ds = SfsmDataset(input_transform=transform)

    elif "amstertime" in dataset_name:
        from dataloaders.val.AmstertimeDataset import AmstertimeDataset
        ds = AmstertimeDataset(input_transform=transform)

    elif "baidu" in dataset_name:
        from dataloaders.val.BaiduDataset import BaiduDataset
        ds = BaiduDataset(input_transform=transform)

    else:
        raise ValueError

    num_references = ds.num_references
    num_queries = ds.num_queries
    ground_truth = ds.ground_truth
    return ds, num_references, num_queries, ground_truth


def get_pca_encoding(model, vlad_encoding):
    pca_encoding = model.WPCA(vlad_encoding.unsqueeze(-1).unsqueeze(-1))
    return pca_encoding


def get_descriptors(model, dataloader, features_path=None):
    descriptors = []
    mapped = None
    offset = 0
    with torch.no_grad():
        with torch.autocast(device_type=args.device, dtype=torch.float16):
            for batch in tqdm(dataloader, "Calculating descritptors..."):
                imgs, labels = batch
                imgs = imgs.to(args.device)

                if args.useFC:  # using fc_output_dim
                    output = model(imgs).cpu()
                else:  # no fc layer after NV layer / vanilla NV
                    if not args.storeSAB and not args.storeSOTL:
                        vlad_encoding = model(imgs)
                    else:
                        image_encoding = model.backbone(imgs)
                        store_path = os.path.join(
                            os.path.dirname(os.path.dirname(args.resume_train)),
                            val_name,
                        )
                        vlad_encoding = model.aggregator(
                            image_encoding,
                            labels=labels.cpu().numpy(),
                            dirPath=store_path,
                        )
                        del image_encoding
                    if args.wpca:
                        vlad_encoding = get_pca_encoding(model, vlad_encoding)
                    output = vlad_encoding.cpu()  # .detach().cpu().numpy()
                if features_path:
                    values = output.float().numpy()
                    if mapped is None:
                        Path(features_path).parent.mkdir(parents=True, exist_ok=True)
                        mapped = np.memmap(features_path, mode="w+", dtype=np.float32,
                                           shape=(len(dataloader.dataset), values.shape[1]))
                    mapped[offset:offset + len(values)] = values
                    offset += len(values)
                else:
                    descriptors.append(output)
                del imgs

    if features_path:
        if mapped is None or offset != len(dataloader.dataset):
            raise RuntimeError("Incomplete evaluation features")
        mapped.flush()
        return mapped
    return torch.cat(descriptors)


def replace_key(k, num):
    for old, new in {"WPCA_" + str(num) + ".": "WPCA."}.items():
        if old in k:
            k = k.replace(old, new)
    return k


def load_model():
    if "netvlad" in args.aggregation.lower():
        agg_config = args
        useToken = False
    elif "salad" in args.aggregation.lower():
        agg_config = {
            "num_channels": args.num_channels,
            "num_clusters": args.num_clusters,
            "cluster_dim": args.cluster_dim,
            "token_dim": args.token_dim,
            "expName": args.expName,
            "reduce_feature_dims": args.reduce_feature_dims,
            "reduce_token_dims": args.reduce_token_dims,
            "args": args,
        }
        useToken = True
    model = VPRModel(
        backbone_arch=args.backbone,
        backbone_config={
            "num_trainable_blocks": args.num_trainable_blocks,
            "return_token": useToken,
            "norm_layer": args.norm_layer,
        },
        agg_arch=args.aggregation,
        agg_config=agg_config,
        args=args,
    )

    if args.ckpt_state_dict:
        checkpoint = torch.load(args.resume_train)
        if args.wpca:
            checkpoint["state_dict"] = OrderedDict(
                {
                    replace_key(k, args.num_pcs): v
                    for k, v in checkpoint["state_dict"].items()
                }
            )
            model_state_dict = model.state_dict()
            # Filter out keys that are not in the model's current state dict
            checkpoint["state_dict"] = {
                k: v
                for k, v in checkpoint["state_dict"].items()
                if k in model_state_dict
            }
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(torch.load(args.resume_train))
    model = model.eval()
    model = model.to(args.device)
    print(f"Loaded model from {args.resume_train} Successfully!")
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval VPR model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_manifest_dir", type=str, default=None,
                        help="Audited manifests; preserve author image order and ground truth")
    parser.add_argument(
        "--dataset_name", type=str, default="gsv_cities", help="Dataset"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./logs/lightning_logs/",
        help="name of directory on which to save the logs, under logs/save_dir",
    )
    parser.add_argument(
        "--expName", default="0", help="Unique string for an experiment"
    )
    parser.add_argument(
        "--batch_size", type=int, default=200, help="Batch size for the data module"
    )
    parser.add_argument("--img_per_place", type=int, default=4, help="Images per place")
    parser.add_argument(
        "--min_img_per_place", type=int, default=4, help="min_img_per_place"
    )
    parser.add_argument(
        "--shuffle_all",
        type=bool,
        default=False,
        help="Shuffle all images or keep shuffling in-city only",
    )
    parser.add_argument(
        "--random_sample_from_each_place",
        type=bool,
        default=True,
        help="Random sample from each place",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[322, 322],
        help="Resizing shape for images (HxW).",
    )

    parser.add_argument("--num_workers", type=int, default=20, help="Number of workers")
    parser.add_argument(
        "--show_data_stats", type=bool, default=True, help="Show data statistics"
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov2_vitb14",
        choices=["dinov2_vitb14", "resnet"],
        help="Backbone architecture",
    )
    parser.add_argument(
        "--num_trainable_blocks", type=int, default=4, help="Trainable blocks"
    )
    parser.add_argument("--norm_layer", type=bool, default=True, help="Use norm layer")

    # Arguments for helper.py (aggregation configuration)
    parser.add_argument(
        "--aggregation",
        type=str,
        default="NETVLAD",
        choices=["SALAD", "cosplace", "gem", "convap", "mixvpr", "NETVLAD"],
    )
    # Cosplace
    parser.add_argument("--in_dim", type=int, default=2048, help="In dim for cosplace")
    parser.add_argument("--out_dim", type=int, default=512, help="In dim for cosplace")
    # gem
    parser.add_argument("--p", type=int, default=3, help="power for gem")
    # convap
    parser.add_argument(
        "--in_channels", type=int, default=2048, help="in_channels for convap"
    )
    # mixvpr
    parser.add_argument(
        "--out_channels", type=int, default=512, help="out_channels for mixvpr"
    )
    parser.add_argument("--in_h", type=int, default=20, help="in_h for mixvpr")
    parser.add_argument("--in_w", type=int, default=20, help="in_w for mixvpr")
    parser.add_argument("--mix_depth", type=int, default=1, help="mix depth for mixvpr")
    # salad
    parser.add_argument(
        "--storeSOTL",
        action="store_true",
        help="Store the soft_assign (optimal transport layer) and vlad",
    )
    parser.add_argument(
        "--num_channels", type=int, default=768, help="num channels for salad"
    )
    parser.add_argument(
        "--num_clusters", type=int, default=64, help="num clusters for salad"
    )
    parser.add_argument(
        "--cluster_dim", type=int, default=128, help="cluster_dim for salad"
    )
    parser.add_argument(
        "--token_dim", type=int, default=256, help="token_dim for salad"
    )
    parser.add_argument(
        "--reduce_feature_dims",
        type=bool,
        default=True,
        help="Perform dimensionlity reduction for feature",
    )

    parser.add_argument(
        "--reduce_token_dims",
        type=bool,
        default=True,
        help="Perform dimensionlity reduction for token",
    )
    # netvlad
    parser.add_argument(
        "--l2",
        type=str,
        default="none",
        choices=["before_pool", "after_pool", "onlyFlatten", "none"],
        help="When (and if) to apply the l2 norm with shallow aggregation layers",
    )
    parser.add_argument(
        "--forLoopAlt",
        type=bool,
        default=True,
        help="if True, it will not use For loop to calculate VLAD ",
    )
    parser.add_argument(
        "--fc_output_dim",
        type=int,
        default=512,
        help="Output dimension of final fully connected layer",
    )
    parser.add_argument("--dim", type=int, default=768, help="dim for netvlad")
    parser.add_argument(
        "--clusters_num", type=int, default=64, help="clusters_num for netvlad"
    )
    parser.add_argument(
        "--initialize_clusters",
        type=bool,
        default=False,
        help="Initialize the cluster for VLAD layer",
    )
    parser.add_argument(
        "--useFC",
        action="store_true",
        help="Add fully connected layer after VLAD layer",
    )
    parser.add_argument(
        "--nv_pca", type=int, help="Use PCA before clustering and nv aggregation."
    )
    parser.add_argument(
        "--nv_pca_randinit",
        action="store_true",
        help="Initialize randomly instead of pca",
    )
    parser.add_argument(
        "--nv_pca_alt", action="store_true", help="use fc layer instead of pca"
    )
    parser.add_argument(
        "--nv_pca_alt_mlp",
        action="store_true",
        help="use 2-fc layer mlp layer instead of pca / pca_alt",
    )

    # ab params
    parser.add_argument(
        "--infer_batch_size",
        type=int,
        default=16,
        help="Batch size for inference (validating and testing)",
    )
    parser.add_argument(
        "--storeSAB",
        action="store_true",
        help="Store the soft_assign, selfDis, w_burst, new soft_assign, vlad",
    )
    parser.add_argument(
        "--antiburst",
        action="store_true",
        help="use self sim + sigmoid to remove burstiness",
    )
    parser.add_argument("--ab_w", type=float, default=8.0, help="")
    parser.add_argument("--ab_b", type=float, default=7.0, help="")
    parser.add_argument("--ab_p", type=float, default=1.0, help="")
    parser.add_argument(
        "--ab_gen", type=int, help="generates thresholds from soft_assign"
    )
    parser.add_argument("--ab_relu", action="store_true", help="")
    parser.add_argument(
        "--ab_soft",
        action="store_true",
        help="softmax instead of sigmoid before summing",
    )
    parser.add_argument("--ab_inv", action="store_true", help="")
    parser.add_argument("--ab_t", type=float, help="thresh for relu")
    parser.add_argument("--ab_testOnly", action="store_true", help="")
    parser.add_argument("--ab_allFreezeButAb", action="store_true", help="")
    parser.add_argument(
        "--ab_fixed",
        action="store_true",
        help="ab params are init but arent nn.Parameter",
    )
    parser.add_argument(
        "--ab_kp", type=int, help="num middle dim for fc-relu-fc weight per pixel"
    )
    parser.add_argument(
        "--ab_wOnly", action="store_true", help="train w, freeze b and p as init"
    )

    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="_"
    )
    parser.add_argument(
        "--resume_train",
        type=str,
        required=True,
        default=None,
        help="path to checkpoint to resume, e.g. logs/.../last_checkpoint.pth",
    )
    parser.add_argument(
        "--ckpt_state_dict", action="store_true", help="Use checkpoint state dictionary"
    )
    # Datasets parameters
    parser.add_argument(
        "--val_datasets",
        nargs="+",
        default=VAL_DATASETS,
        help="Validation datasets to use",
        choices=VAL_DATASETS,
    )
    parser.add_argument(
        "--pl_seed",
        type=bool,
        default=True,
        help="Use pytorch_lightning (pl) seed",
    )
    parser.add_argument(
        "--wpca",
        action="store_true",
        help="Use post pool WPCA layer",
    )
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--num_pcs", type=int, default=8192, help="Use post pool PCA.")
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Dont use wandb",
    )
    parser.add_argument(
        "--store_eval_output", action="store_true", help="store eval output"
    )

    args = parser.parse_args()

    # Parse image size
    if args.resize:
        if len(args.resize) == 1:
            args.resize = (args.resize[0], args.resize[0])
        elif len(args.resize) == 2:
            args.resize = tuple(args.resize)
        else:
            raise ValueError("Invalid image size, must be int, tuple or None")

        args.resize = tuple(map(int, args.resize))

    return args


if __name__ == "__main__":
    args = parse_args()
    from utils.resource_budget import configure_cuda_budget
    configure_cuda_budget(args.device)

    # Check: dont have wpca for systems having fc layer after NV layer
    if args.useFC:
        assert args.wpca == False

    if not args.no_wandb:
        dataset_name = args.dataset_name.lower()[:4]
        wandb_dataStr = dataset_name
        args.expName = "eval-" + wandb_dataStr + args.expName

    if args.pl_seed:
        pl.seed_everything(seed=int(args.seed), workers=True)

    model = load_model()
    checkpoint_hash = hashlib.sha256()
    with open(args.resume_train, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checkpoint_hash.update(block)
    print(f"<=======backbone_arch: {args.backbone}========>")
    print(f"<=======aggregation: {args.aggregation}========>")
    print(f"<=======args: {args}========>")
    print(f"<=======model: {model}========>")

    if not args.no_wandb:
        wandb.init(project="vlad_buff", config=args)
        # update runName
        runName = wandb.run.name
        wandb.run.name = args.expName + "-" + runName.split("-")[-1]
        wandb.run.save()

    for val_name in args.val_datasets:
        print(f"<=======expName: {args.expName}========>")

        val_dataset, num_references, num_queries, ground_truth = get_val_dataset(
            val_name, args.resize, args.dataset_manifest_dir
        )
        val_loader = DataLoader(
            val_dataset,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            **({"prefetch_factor": 1} if args.num_workers else {}),
        )
        print(f"Evaluating on {val_name}")
        features_path = Path(args.save_dir) / "working_features" / f"{val_name}.dat"
        descriptors = get_descriptors(model, val_loader, features_path=features_path)

        print(f"Descriptor dimension {descriptors.shape[1]}")
        r_list = descriptors[:num_references]
        q_list = descriptors[num_references:]

        print("total_size", descriptors.shape[0], num_queries + num_references)
        print(f"Queries:{num_queries}, References:{num_references}")

        distances, predictions, preds = get_validation_recalls(
            r_list=r_list,
            q_list=q_list,
            k_values=[1, 5, 10] if ground_truth is not None else [1, 5, 10, 25],
            gt=ground_truth,
            print_results=True,
            dataset_name=val_name,
            faiss_gpu=False,
            testing=False,
            retrieval_batch_size=256,
        )

        if args.store_eval_output:
            os.makedirs(args.save_dir, exist_ok=True)
            if args.resume_train is not None:
                model_ckpt_name = os.path.basename(args.resume_train)
            else:
                model_ckpt_name = "_"
            np.savez(
                os.path.join(
                    args.save_dir, model_ckpt_name + "_" + val_name + "_predictions.npz"
                ),
                predictions=predictions,
                distances=distances,
                query_ids=np.asarray(getattr(val_dataset, "query_ids", [])),
                reference_ids=np.asarray(getattr(val_dataset, "reference_ids", [])),
            )
            metrics_path = Path(args.save_dir) / (model_ckpt_name + "_" + val_name + "_metrics.json")
            with open(metrics_path.with_suffix('.tmp'), "w") as f:
                json.dump({
                    "checkpoint": os.path.abspath(args.resume_train),
                    "checkpoint_sha256": checkpoint_hash.hexdigest(),
                    "dataset": val_name,
                    "num_references": num_references,
                    "num_queries": num_queries,
                    "descriptor_dim": descriptors.shape[1],
                    "resize": args.resize,
                    "recalls": None if preds is None else {str(k): float(v) for k, v in preds.items()},
                    "ground_truth_available": ground_truth is not None,
                    "recall_denominator": num_queries if ground_truth is not None else None,
                    "queries_without_positives": None if ground_truth is None else sum(len(v) == 0 for v in ground_truth),
                    "protocol": getattr(val_dataset, "protocol", {"name": "legacy author loader"}),
                    "protocol_fingerprint": getattr(val_dataset, "protocol_fingerprint", None),
                    "manifest_sha256": getattr(val_dataset, "manifest_sha256", None),
                    "manifest_path": str(getattr(val_dataset, "manifest_path", "")),
                    "inference_precision": "float16-autocast",
                    "retrieval": "CPU IndexFlatL2 float32; chunks of 256",
                    "preprocessing": "one descriptor per image; bilinear Resize(H,W), ImageNet normalization",
                }, f, indent=2)
            metrics_path.with_suffix('.tmp').replace(metrics_path)
            if ground_truth is None and hasattr(val_dataset, "save_predictions"):
                val_dataset.save_predictions(predictions, str(Path(args.save_dir) / f"{val_name}_top25.txt"))
        if not args.no_wandb and preds is not None:
            metrics = {
                f"{val_name}/R@1VsDim": {
                    "Recall@1": preds[1],
                    "Dim": descriptors.shape[1],
                }
            }
            wandb.log(metrics)

            metrics = {f"{val_name}/Recall@{k}": v for k, v in preds.items()}
            wandb.log(metrics)

        del r_list, q_list
        descriptors._mmap.close()
        features_path.unlink()
        del descriptors
