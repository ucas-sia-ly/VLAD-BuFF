import numpy as np
import faiss
import faiss.contrib.torch_utils
from prettytable import PrettyTable


def get_validation_recalls(
    r_list,
    q_list,
    k_values,
    gt,
    print_results=True,
    faiss_gpu=False,
    dataset_name="dataset without name ?",
    testing=False,
    retrieval_batch_size=None,
):
    # faiss.contrib.torch_utils checks exact ndarray type; memmap is a subclass.
    # np.asarray creates a view of mapped storage without reading/copying it all.
    if isinstance(r_list, np.ndarray):
        r_list = np.asarray(r_list)
    if isinstance(q_list, np.ndarray):
        q_list = np.asarray(q_list)
    embed_size = r_list.shape[1]
    if faiss_gpu:
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = True
        flat_config.device = 0
        faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)
    # build index
    else:
        faiss_index = faiss.IndexFlatL2(embed_size)

    # add references
    if retrieval_batch_size is None:
        faiss_index.add(r_list)
        distances, predictions = faiss_index.search(q_list, max(k_values))
    else:
        if retrieval_batch_size <= 0:
            raise ValueError("retrieval_batch_size must be positive")
        for start in range(0, len(r_list), retrieval_batch_size):
            faiss_index.add(r_list[start:start + retrieval_batch_size])
        distances = np.empty((len(q_list), max(k_values)), dtype=np.float32)
        predictions = np.empty((len(q_list), max(k_values)), dtype=np.int64)
        for start in range(0, len(q_list), retrieval_batch_size):
            stop = min(start + retrieval_batch_size, len(q_list))
            distances[start:stop], predictions[start:stop] = faiss_index.search(q_list[start:stop], max(k_values))

    if testing:
        return predictions
    if gt is None:
        if print_results:
            print(f"{dataset_name}: no public ground truth; predictions only, Recall unavailable")
        return distances, predictions, None

    # start calculating recall_at_k
    correct_at_k = np.zeros(len(k_values))

    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[q_idx])):
                correct_at_k[i:] += 1
                break

    correct_at_k = correct_at_k / len(predictions)
    recalls = {k: v for (k, v) in zip(k_values, correct_at_k)}

    if print_results:
        print()  # print a new line
        table = PrettyTable()
        table.field_names = ["K"] + [str(k) for k in k_values]
        table.add_row(["Recall@K"] + [f"{100*v:.2f}" for v in correct_at_k])
        print(table.get_string(title=f"Performances on {dataset_name}"))

    return distances, predictions, recalls
