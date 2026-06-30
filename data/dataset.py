"""Pre-training dataset with document packing."""
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer


class PretrainDataset(Dataset):
    """Dataset that tokenizes documents and packs them into fixed-length sequences."""

    def __init__(self, file_path: str, tokenizer_path: str, max_seq_len: int):
        self.max_seq_len = max_seq_len
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        all_tokens = []
        bos_id = self.tokenizer.token_to_id("<s>")
        eos_id = self.tokenizer.token_to_id("</s>")
        for line in lines:  # noqa
            tokens = self.tokenizer.encode(line).ids
            all_tokens.extend([bos_id] + tokens + [eos_id])

        total_len = (len(all_tokens) // max_seq_len) * max_seq_len
        self.data = torch.tensor(all_tokens[:total_len], dtype=torch.long).view(-1, max_seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx]
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def create_dataloader(
    file_path: str,
    tokenizer_path: str,
    max_seq_len: int,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle: bool = True,
) -> DataLoader:
    dataset = PretrainDataset(file_path, tokenizer_path, max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
