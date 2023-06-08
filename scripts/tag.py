from argparse import ArgumentParser, Namespace
from pathlib import Path

import spacy
import torch
from datasets import Dataset, load_dataset
from torch.multiprocessing import spawn
from tqdm.auto import tqdm

from utils import assert_type


def worker(rank: int, args: Namespace):
    world_size = torch.cuda.device_count()

    ds = (
        Dataset.from_json(args.dataset)
        if args.dataset.endswith(".json")
        else load_dataset(args.dataset, split="train")
    )
    dataset = assert_type(Dataset, ds).filter(
        lambda x: len(x[args.text_column]) < args.max_doc_length
    ).shard(world_size, rank)

    # Make our own directory for the output
    out_dir = args.out_dir / f"rank_{rank}"
    out_dir.mkdir(parents=True, exist_ok=True)

    spacy.require_gpu(gpu_id=rank)
    nlp = spacy.load(
        "en_core_web_trf",
        # Important for performance; otherwise Tensor Cores aren't used
        config=dict(
            components=dict(
                transformer=dict(model=dict(mixed_precision=True))
            )
        ),
    )

    doc_iter = nlp.pipe(
        map(lambda x: x[args.text_column], dataset),
        batch_size=args.batch_size,
    )

    for i, doc in tqdm(
        enumerate(doc_iter), position=rank, smoothing=0.0, total=len(dataset)
    ):
        # This is a little sketchy and depends how Dataset.shard works
        doc_id = rank + i * world_size
        doc.to_disk(out_dir / f"doc_{doc_id}.spacy", exclude=["tensor", "user_data"])


if __name__ == "__main__":
    parser = ArgumentParser(description="Tag a dataset with Spacy")
    parser.add_argument(
        "dataset",
        type=str,
        help="Either a path to a JSON file or a HuggingFace dataset name",
    )
    parser.add_argument(
        "out_dir",
        type=Path,
        default=Path("data/pile-spacy"),
        help="Output directory"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for tagger model",
    )
    parser.add_argument(
        "--max-doc-length",
        type=int,
        default=500_000,
        help="Documents with more than this many characters are dropped."
    )
    parser.add_argument(
        "--text-column", type=str, default="text", help="Column name for text"
    )
    args = parser.parse_args()

    spawn(
        worker,
        args=(args,),
        nprocs=torch.cuda.device_count(),
        join=True,
    )
