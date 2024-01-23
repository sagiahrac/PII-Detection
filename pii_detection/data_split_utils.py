import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _save_split_shards(train, val, test, save_dir):
    for name, dataset in zip(["train", "val", "test"], [train, val, test]):
        dst = Path(save_dir) / f"{name}_shard.json"
        with dst.open("w") as f:
            json.dump(dataset, f)


def shuffle_and_split(
    data: List[Dict[str, Any]],
    ratios: List[float] = [0.7, 0.2, 0.1],
    save_dir: str = None,
    seed: int = 42,
):
    assert np.allclose(sum(ratios), 1)
    data = sorted(data, key=lambda x: x["document"])
    np.random.seed(seed)
    np.random.shuffle(data)

    N = len(data)
    train_end_idx = int(N * ratios[0])
    val_end_idx = int(N * sum(ratios[:2]))

    train = data[:train_end_idx]
    val = data[train_end_idx:val_end_idx]
    test = data[val_end_idx:]
    assert len(train) + len(val) + len(test) == N

    if save_dir:
        _save_split_shards(train, val, test, save_dir)
    return train, val, test


def get_labels_count(train, val, test) -> pd.DataFrame:
    counts_list = []
    for dataset in [train, val, test]:
        labels = [essay["labels"] for essay in dataset]
        labels_unnested = [label for sublist in labels for label in sublist]
        counts = pd.Series(labels_unnested).value_counts()
        counts_list.append(counts)

    counts_df = pd.concat(counts_list, axis=1).fillna(0)
    counts_df.columns = ["train", "val", "test"]
    return counts_df


def map_labels_to_essays(data: List[Dict[str, Any]]) -> pd.Series:
    labels_unique = set(label for sublist in data for label in sublist["labels"])
    labels_unique_dict = {label: [] for label in labels_unique}

    for essay in data:
        essay_labels = set(essay["labels"])
        for label in labels_unique:
            if label in essay_labels:
                labels_unique_dict[label].append(essay["document"])

    return pd.Series(labels_unique_dict, name="essay_ids")


def get_essays_with_rare_labels(
    labels_to_essays: pd.Series, threshold: int = 5
) -> List[str]:
    essay_count = labels_to_essays.apply(len)
    essay_ids = labels_to_essays[essay_count <= threshold]
    return essay_ids


def split_rare_essays_from_data(data: List[Dict[str, Any]], essay_ids: List[str]):
    essays_to_remove = []
    for essay in data:
        if essay["document"] in essay_ids:
            essays_to_remove.append(essay)

    essays_to_keep = [essay for essay in data if essay not in essays_to_remove]
    return essays_to_keep, essays_to_remove
