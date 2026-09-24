import sys
import open_clip
import torch
import numpy as np
import autokeras as ak

from PIL import Image
from tensorflow.keras.models import load_model as keras_load_model

from models.NSFWModelWrapper import NSFWModelWrapper  # noqa: E402


class LaionClipDetector(NSFWModelWrapper):
    name = "clip_laion_probed"

    def __init__(self, 
        mlp_head_path: str,
        clip_model_name: str = "ViT-L-14",
        clip_pretrained: str = "openai", 
        device: str = "cpu"
    ):
        self.device = device
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained
        )
        self.clip_model = self.clip_model.to(device).eval()
        self.mlp_head = keras_load_model(mlp_head_path, custom_objects=ak.CUSTOM_OBJECTS, compile=False)
    
    def _preprocess(self, filepath: str) -> None:
        self.image = Image.open(filepath).convert("RGB")

    def _embed(self) -> np.ndarray:
        with torch.no_grad():
            x = self.preprocess(self.image).unsqueeze(0).to(self.device)
            embedding = self.clip_model.encode_image(x)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # normalizar,
            # tal como lo espera la cabeza MLP entrenada por LAION-AI
        return embedding.cpu().numpy()

    def _predict_probability(self) -> float:
        if self.image is None:
            raise RuntimeError(f"{self.name} trying to run without an image configured, run preprocess(image)")

        embedding = self._embed()
        score = self.mlp_head.predict(embedding, verbose=0)[0]
        # el modelo de LAION regresa [P(safe), P(unsafe)] o un solo score,
        # según la versión descargada -- VERIFICAR contra la salida real
        # la primera vez que se corra, y ajustar el índice si es necesario
        return float(score[0]) if hasattr(score, "__len__") else float(score)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp-head", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    model = LaionClipDetector(args.mlp_head)
    pred = model.predict(args.image)
    print(pred)
