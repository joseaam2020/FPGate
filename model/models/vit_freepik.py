import sys
import torch

from PIL import Image
from pathlib import Path
from timm.data import resolve_data_config
from torchvision.transforms import Compose
from timm.models import get_pretrained_cfg
from timm.data.transforms_factory import create_transform
from transformers import AutoImageProcessor, AutoModelForImageClassification

from models.NSFWModelWrapper import NSFWModelWrapper

idx_to_label = {0: 'neutral', 1: 'low', 2: 'medium', 3: 'high'}

class FreepikEva02(NSFWModelWrapper):
    name = "vit_freepik_eva02"

    def __init__(self, model_id: str = "Freepik/nsfw_image_detector"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model  = AutoModelForImageClassification.from_pretrained(model_id, torch_dtype=dtype).to(self.device)
        cfg = get_pretrained_cfg("eva02_base_patch14_448.mim_in22k_ft_in22k_in1k")
        self.processor: Compose = create_transform(**resolve_data_config(cfg.__dict__))

    def _preprocess(self, filepath: str) -> None:
        self.image = Image.open(filepath).convert("RGB")

    def _predict_probability(self) -> float:
        if self.image is None:
            raise RuntimeError(f"{self.name} trying to run without an image configured, run preprocess(image)")

        inputs = self.processor(self.image).unsqueeze(0).to(self.device, dtype=self.model.dtype)
        with torch.inference_mode():
            probs = torch.softmax(self.model(inputs).logits.float(), dim=-1)[0]
        print(probs)
        return probs[1:].sum().item()
            

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    model = FreepikEva02()
    pred = model.predict(args.image)
    print(pred)
