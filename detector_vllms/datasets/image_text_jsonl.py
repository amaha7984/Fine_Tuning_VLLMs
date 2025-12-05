from pathlib import Path
import json

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class ImageTextJsonlDataset(Dataset):
    """
    Generic image+text dataset for AI-generation detection.

    Expected JSONL keys:
      - image_path: path to image relative to jsonl or absolute
      - question: user question / prompt
      - answer: ground-truth explanation text (for SFT)
      - label: 'real' or 'fake' (optional but useful)
    """

    def __init__(self, jsonl_path: str | Path, image_root: str | Path | None = None, transform=None):
        self.jsonl_path = Path(jsonl_path)
        self.image_root = Path(image_root) if image_root is not None else self.jsonl_path.parent
        self.items = []

        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))

        self.transform = transform or T.Compose(
            [
                T.Resize((256, 256)), 
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = self.image_root / item["image_path"]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)

        return {
            "image": image_tensor,
            "image_path": str(img_path),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "label": item.get("label", None),
        }
