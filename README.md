# Part-of-Speech Tagging for the Pile and RedPajama

This repository contains the code for generating the part-of-speech tagged Pile and RedPajama corpora used in the paper [LEACE: Perfect linear concept erasure in closed form](https://arxiv.org/abs/2306.03819).

## Usage

The code is written in Python 3.10. To install the dependencies, run:

```bash
pip install -r requirements.txt
```

Generating the dataset is a two-step process.

1. The script `tag.py` runs the SpaCy model `en_core_web_trf` on the specified dataset and saves the raw SpaCy `Doc` objects to a folder. This step is independent of the model you plan to feed the dataset into.
2. The script `aligned_tokenize.py` loads the SpaCy `Doc` objects and tokenizes them with a pretrained HuggingFace `transformers` tokenizer, taking care to ensure that the part-of-speech tags are aligned with token boundaries in the most reasonable way possible. It also breaks the documents into 2048-token chunks, concatenating chunks together (delimited by EOS tokens) when necessary to keep the length of each chunk the same. It then saves a HuggingFace `datasets` file containing the tokenized dataset.

## Example

The `tag.py` script will accept either a path to a `.jsonl` file containing the dataset, or the name of a dataset on the HuggingFace Hub. In the paper, we used the _validation_ set of The Pile, and the `togethercomputer/RedPajama-Data-1T-Sample` sample from the LLaMA pretraining corpus. Unfortunately, it's currently not possible to download The Pile validation set from HuggingFace without downloading the train set as well (which weighs hundreds of gigabytes), so we recommend directly downloading the compressed `.jsonl` file:
```bash
wget https://the-eye.eu/public/AI/pile/val.jsonl.zst
```

You'll need to have the [Zstandard (de)compression tool](https://github.com/facebook/zstd) installed to decompress the file:
```bash
zstd -d val.jsonl.zst
```

If you haven't used SpaCy before, you'll probably need to explicitly download the relevant model:
```bash
python -m spacy download en_core_web_trf
```

You can then run `tag.py` on the file:
```bash
python scripts/tag.py val.jsonl <output-dir>
```

Followed by `aligned_tokenize.py`, with the name of the tokenizer of your choice:
```bash
python scripts/aligned_tokenize.py EleutherAI/pythia-160m val.jsonl <output-dir2>
```