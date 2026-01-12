import os
import json
from typing import Optional, Dict, Any

from torch.utils.data import Dataset

# ============================================================
# Fallacy label definitions
# ============================================================

# You can adjust this list according to the actual CoCoLoFa labels.
# The last label "none" means "no fallacy".
FALLACY_LABELS = [
    "appeal to authority",
    "appeal to majority",
    "appeal to nature",
    "appeal to tradition",
    "appeal to worse problems",
    "false dilemma",
    "hasty generalization",
    "slippery slope",
    "none",
]

LABEL2ID: Dict[str, int] = {lbl: i for i, lbl in enumerate(FALLACY_LABELS)}
ID2LABEL: Dict[int, str] = {i: lbl for lbl, i in LABEL2ID.items()}


# ============================================================
# CoCoLoFa dataset wrapper
# ============================================================

class CoCoLoFaDataset(Dataset):
    """
    PyTorch Dataset for the CoCoLoFa logical fallacy dataset.

    The dataset is assumed to be stored in a JSON file with a structure like:
        [
          {
            "id": 427,
            "title": "...",
            "content": "...",
            "comments": [
              {
                "id": "6078",
                "fallacy": "slippery slope",
                "comment": "actual comment text here"
              },
              ...
            ]
          },
          ...
        ]

    Each comment is treated as one training example:
        - text: comment text (optionally with article content prepended)
        - label: index of the fallacy type (including "none")
    """

    def __init__(
        self,
        path: str,
        split: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        max_length: int = 512,
        use_article: bool = False,
        drop_unknown_label: bool = False,
    ):
        """
        Args:
            path:
                Either a JSON file path or a directory containing JSON files.
                If it is a directory and `split` is not None, we will load
                `os.path.join(path, f"{split}.json")`.
            split:
                Optional split name, e.g. "train", "dev", "test".
                Used only when `path` is a directory.
            tokenizer:
                HuggingFace tokenizer or any callable with the same interface.
                If provided, __getitem__ returns tokenized tensors.
                If None, __getitem__ returns raw text + label.
            max_length:
                Max sequence length for tokenization.
            use_article:
                If True, prepend the article content before the comment:
                    text = article_content + "\n\n" + comment
                If False, only use the comment as input.
            drop_unknown_label:
                If True, skip samples whose `fallacy` is not in FALLACY_LABELS.
                If False, map unknown labels to "none".
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_article = use_article
        self.drop_unknown_label = drop_unknown_label

        # Resolve actual JSON file path
        if os.path.isdir(path):
            if split is None:
                raise ValueError(
                    "When `path` is a directory, you must provide `split`, "
                    "so we can load `f'{split}.json'` inside that directory."
                )
            file_path = os.path.join(path, f"{split}.json")
        else:
            # Assume `path` is already a file path
            file_path = path

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"CoCoLoFa JSON file not found: {file_path}")

        # ------------------------------------------------------------
        # Load JSON and convert article-level structure to comment-level samples
        # ------------------------------------------------------------
        with open(file_path, "r", encoding="utf-8") as f:
            articles = json.load(f)

        self.samples = []
        for art in articles:
            article_text = art.get("content", "") or ""
            comments = art.get("comments", [])
            for c in comments:
                comment_text = c.get("comment", "")
                label_str = c.get("fallacy", "")

                # Handle labels: drop unknown or map to "none"
                if label_str not in LABEL2ID:
                    if self.drop_unknown_label:
                        continue
                    else:
                        label_str = "none"  # treat as no fallacy

                if self.use_article:
                    # Prepend article content
                    text = article_text + "\n\n" + comment_text
                else:
                    text = comment_text

                self.samples.append(
                    {
                        "text": text,
                        "label": LABEL2ID[label_str],
                    }
                )

        if len(self.samples) == 0:
            raise ValueError(
                f"No samples loaded from {file_path}. "
                f"Check the JSON structure and label names."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        text = item["text"]
        label = item["label"]

        # If no tokenizer is provided, return raw text + label
        if self.tokenizer is None:
            return {
                "text": text,
                "label": label,
            }

        # Otherwise, return tokenized tensors that Trainer / DataLoader can use
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # tokenizer returns batch dimension of size 1; remove it
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        encoded["labels"] = label
        # assert isinstance(label_id, int), f"label_id not int: {label_id} ({type(label_id)})"
        # assert 0 <= label_id < num_labels, f"label_id out of range: {label_id}"
        return encoded

