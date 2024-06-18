import torch
import random
import numpy as np


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]  # [b, n_e*(n_e-1), n_rel]
    entity_pos = [f["entity_pos"] for f in batch]  # [b, n_e, n_e_m]
    hts = [f["hts"] for f in batch]  # [b, n_e*(n_e-1), 2]
    titles = [f["title"] for f in batch]  # [b]
    input_ids = torch.tensor(input_ids, dtype=torch.long)  # [b, max]
    input_mask = torch.tensor(input_mask, dtype=torch.float)  # [b, max]
    output = (input_ids, input_mask, labels, entity_pos, hts, titles)
    return output
