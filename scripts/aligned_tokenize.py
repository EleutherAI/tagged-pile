import os
from functools import partial
from itertools import chain
from multiprocessing import cpu_count, Pool
from pathlib import Path

import spacy
from datasets import (
    ClassLabel,
    Dataset,
    Features,
    Sequence,
    Value,
    concatenate_datasets,
)
from spacy.tokens import Doc
from transformers import AutoTokenizer

from utils import assert_type


def spacy_to_hf(spacy_dir: Path, tokenizer_name: str):
    # Make sure that the tokenizer doesn't use multiple threads
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, model_max_length=None,
    )
    vocab = spacy.load("en_core_web_trf").vocab

    # Load the spacy docs
    for path in spacy_dir.rglob("*.spacy"):
        doc = Doc(vocab).from_disk(path)

        encoding = tokenizer(
            doc.text,
            max_length=None,
            return_offsets_mapping=True,
        )

        # In general, BPE tokens will be subspans of SpaCy tokens. Each BPE token
        # should get its label from the SpaCy token that contains it.
        spacy_spans = [
            doc.char_span(*span, alignment_mode="expand")
            for span in encoding.pop("offset_mapping")
        ]
        # In rare cases, the BPE token may be split across multiple SpaCy tokens. In
        # those cases, we assign the token to the first SpaCy token that contains it.
        parts_of_speech = [
            # Whitespace might get tokenized into a BPE token, but it doesn't have
            # a part of speech. Assign it the "X" (other) tag.
            span[0].pos_ if span and span[0].pos_ != "SPACE" else "X"
            for span in spacy_spans
        ]
        yield dict(
            input_ids=encoding["input_ids"],
            pos=parts_of_speech,
        )


def process_shard(rank_path: Path, tokenizer_name: str) -> Dataset:
    ds = Dataset.from_generator(
        spacy_to_hf,
        features=Features(
            {
                "input_ids": Sequence(Value(dtype="int32")),
                "pos": Sequence(
                    # Universal Dependency POS tags
                    ClassLabel(
                        names=[
                            "ADJ",
                            "ADP",
                            "ADV",
                            "AUX",
                            "CCONJ",
                            "DET",
                            "INTJ",
                            "NOUN",
                            "NUM",
                            "PART",
                            "PRON",
                            "PROPN",
                            "PUNCT",
                            "SCONJ",
                            "SYM",
                            "VERB",
                            "X",
                        ]
                    ),
                ),
            }
        ),
        gen_kwargs=dict(spacy_dir=rank_path, tokenizer_name=tokenizer_name),
    )
    return assert_type(Dataset, ds)


def chunk(seq: list[int], chunk_size: int) -> list[list[int]]:
    """Chunk a sequence into chunks of size `chunk_size`."""

    return [
        seq[i * chunk_size : (i + 1) * chunk_size]
        for i in range(len(seq) // chunk_size)
    ]


def uniform_chunks(
    batch: dict[str, list], eos_id: int, eos_pos: int, chunk_size: int
) -> dict[str, list]:
    pos_iter = chain.from_iterable(ids + [eos_pos] for ids in batch["pos"])
    token_iter = chain.from_iterable(
        ids + [eos_id] for ids in batch["input_ids"]
    )
    return {
        "input_ids": chunk(list(token_iter), chunk_size),
        "pos": chunk(list(pos_iter), chunk_size),
    }


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Tokenize SpaCy docs while aligning the part-of-speech tags"
    )
    parser.add_argument("tokenizer_name", type=str)
    parser.add_argument("spacy_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--chunk-size", type=int, default=2048)
    args = parser.parse_args()

    folders = list(args.spacy_dir.glob("rank_*"))
    process_fn = partial(process_shard, tokenizer_name=args.tokenizer_name)

    with Pool(len(folders)) as pool:
        shards = pool.map(process_fn, folders)
        master = concatenate_datasets(shards)

    if args.chunk_size:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        print(f"Breaking into uniform chunks of size {args.chunk_size}...")

        master = master.map(
            uniform_chunks,
            batched=True,
            batch_size=1000,
            fn_kwargs=dict(
                eos_id=tokenizer.eos_token_id,
                eos_pos=master.features["pos"].feature.str2int("X"),
                chunk_size=args.chunk_size,
            ),
            num_proc=cpu_count() // 2,
        ).map(
            lambda x: {
                "num_bytes": len(tokenizer.decode(x["input_ids"]).encode("utf-8"))
            },
            num_proc=cpu_count() // 2,
        )

    # Save the dataset
    master.save_to_disk(args.output_dir)
