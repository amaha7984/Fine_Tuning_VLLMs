from pathlib import Path
import json

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class ExplanationOnlyDataset(Dataset):
    """
    Text-only variant (previous version, kept for reference).
    Returns image tensor (not used in text-only training), question, answer.
    """

    def __init__(self, root: str | Path, split: str = "train", transform=None):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.ann_path = self.root / "annotations" / f"{split}.json"

        with self.ann_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        self.items = []
        for obj in raw:
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


class ExplanationOnlyVLDataset(Dataset):
    """
    VLM-friendly variant.

    Returns:
      - PIL image (no manual transform; let the Qwen-VL processor handle it)
      - question: fixed instruction-like query
      - answer: caption (explanation)
    """

    def __init__(self, root: str | Path, split: str = "train"):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.ann_path = self.root / "annotations" / f"{split}.json"

        with self.ann_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        self.items = []
        for obj in raw:
            for _, entry in obj.items():
                self.items.append(
                    {
                        "img_file_name": entry["img_file_name"],
                        "caption": entry["caption"],
                    }
                )

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
        image = image.resize((384, 384)) 

        question = self.question_template
        answer = item["caption"]

        return {
            "image": image,               # PIL image for processor
            "image_path": str(img_path),
            "question": question,
            "answer": answer,
        }
