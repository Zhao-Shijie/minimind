"""Text generation from a trained MiniMind checkpoint."""
import argparse

import torch

from model.model import MiniMind, ModelConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with MiniMind")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    model_cfg = ModelConfig(**cfg.get("model", {}))
    tokenizer_path = cfg.get("data", {}).get("tokenizer_path", "data/tokenizer.json")

    model = MiniMind(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Tokenize prompt
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    input_ids = torch.tensor([tokenizer.encode(args.prompt).ids], device=device)

    print(f"Prompt: {args.prompt}")
    print("=" * 60)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.token_to_id("</s>"),
    )

    generated = tokenizer.decode(output_ids[0].tolist())
    print(generated)


if __name__ == "__main__":
    main()
