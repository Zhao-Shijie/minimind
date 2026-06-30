"""Train a BPE tokenizer using HuggingFace tokenizers library."""
import argparse
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


def train_tokenizer(
    files: list,
    vocab_size: int = 32768,
    min_frequency: int = 2,
    output_path: str = "data/tokenizer.json",
):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<s>", "</s>", "<unk>", "<pad>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train(files, trainer)
    tokenizer.save(output_path)
    print(f"Tokenizer saved to {output_path} (vocab size: {tokenizer.get_vocab_size()})")
    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--vocab_size", type=int, default=32768)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--output", default="data/tokenizer.json")
    args = parser.parse_args()

    for f in args.files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"File not found: {f}")

    train_tokenizer(args.files, args.vocab_size, args.min_frequency, args.output)
