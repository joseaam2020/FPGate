import sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from pathlib import Path
from split.to_binary import to_binary  # noqa: E402
from models.NSFWModelWrapper import NSFWModelWrapper  # noqa: E402


# Map the target explicit class indices once at the class level or module scope
# GANTMAN_CLASSES = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']
EXPLICIT_INDICES = {1, 3, 4}  # 'hentai': 1, 'porn': 3, 'sexy': 4


class GantManCNN(NSFWModelWrapper):

    name = "cnn_gantman_mobilenetv2"
    image = None 

    def __init__(self, model_path_str: str, input_size: int = 224):
        model_path = Path(model_path_str)

        if not model_path.exists():
            raise ValueError("saved_model_path must be the valid directory of a saved model to load.")
        
        self.model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer},compile=False)
        self.input_size = input_size

    def _preprocess(self, filepath: str) -> None:
        self.image = None

        try:
            image = tf.keras.preprocessing.image.load_img(filepath, target_size=(self.input_size, self.input_size))
            image = tf.keras.preprocessing.image.img_to_array(image)
            image /= 255
            self.current_filepath = filepath
            self.image = np.expand_dims(image, axis=0)

        except Exception as ex:
            print("Image Load Failure: ", filepath, ex)

    def _predict_probability(self) -> float:
        if self.image is None:
            raise RuntimeError(f"{self.name} trying to run without an image configured, run preprocess(image)")

        model_preds = self.model.predict(self.image, verbose=0)
        
        preds = model_preds[0] if model_preds.ndim > 1 else model_preds
        print(preds)

        return float(preds[list(EXPLICIT_INDICES)].sum())


if __name__ == "__main__":
    # Prueba rápida manual
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    model = GantManCNN(args.weights)
    pred = model.predict(args.image)
    print(pred)
