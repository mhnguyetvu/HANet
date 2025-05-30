import torch
import json
import os
import random
from collections import defaultdict
from typing import Dict, List


def build_memory_set(dataset, label2id, k=1):
    """
    Select k exemplar instances per class based on closest to prototype (mean representation).
    Args:
        dataset: a list of examples, each with keys: 'input_ids', 'attention_mask', 'trigger_mask', 'labels'
        label2id: dictionary of class labels
        k: number of exemplars per class
    Returns:
        memory: Dict[label_id] = list of exemplar inputs (dictionaries)
    """
    memory = defaultdict(list)
    by_class = defaultdict(list)

    for ex in dataset:
        lbl = ex['labels'].item() if isinstance(ex['labels'], torch.Tensor) else ex['labels']
        by_class[lbl].append(ex)

    for lbl, examples in by_class.items():
        chosen = random.sample(examples, min(k, len(examples)))
        memory[lbl].extend(chosen)

    return dict(memory)


def save_memory(memory: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(memory, path)


def load_memory(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    return torch.load(path)


def prototypical_augment(memory: Dict[int, List[Dict]], num_aug=5, stddev=0.1):
    """
    Generate synthetic trigger representations from exemplars using Gaussian noise.
    Args:
        memory: Dict[label_id] = list of dicts (each with 'repr')
        num_aug: number of augmentations per exemplar
        stddev: standard deviation for Gaussian noise
    Returns:
        List of tuples: (augmented_repr, label_id)
    """
    aug_repr = []
    for label, ex_list in memory.items():
        for ex in ex_list:
            if 'repr' not in ex:
                continue
            base = ex['repr']
            noise = torch.randn((num_aug, base.shape[-1])) * stddev
            samples = base.unsqueeze(0) + noise
            for s in samples:
                aug_repr.append((s, label))
    return aug_repr
