from pathlib import Path
import json

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class ExplanationOnlyDataset(Dataset):
    """
    Dataset for explanation-only training on (likely) fake images.

    Expected structure:
      root/
        images/
        annotations/train.json

    train.json format (simplified):
      [
        {
          "0": {
            "caption": "...",
            "img_file_name": "...",
            "refs": [...]
          }
        },
        {
          "1": {
            "caption": "...",
            "img_file_name": "...",
            "refs": [...]
          }
        },
        ...
      ]

    We only use img_file_name + caption.
    """

    def __init__(self, root: str | Path, split: str = "train", transform=None):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.ann_path = self.root / "annotations" / f"{split}.json"

        with self.ann_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        # Flatten into a simple list of {img_file_name, caption}
        self.items = []
        for obj in raw:
            # each obj is like {"0": {...}} or {"123": {...}}
            for _, entry in obj.items():
                self.items.append(
                    {
                        "img_file_name": entry["img_file_name"],
                        "caption": entry["caption"],
                    }
                )

        self.transform = transform or T.Compose(
            [
                T.Resize((448, 448)),
                T.ToTensor(),
            ]
        )

        # fixed question for training
        self.question_template = (
            "Describe any artifacts or visual clues that suggest this image "
            "may be AI-generated. Be specific and detailed."
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img_path = self.images_dir / item["img_file_name"]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)

        question = self.question_template
        answer = item["caption"]

        return {
            "image": image_tensor,
            "image_path": str(img_path),
            "question": question,
            "answer": answer,
        }