"""
Corre GantMan y LAION (los 2 modelos basados en TF/Keras) sobre el test
split congelado, y escribe predictions_tf.csv.

Correr dentro de env_tf:
    python run_eval_tf.py --test splits/test.csv \
        --gantman-weights path/al/mobilenet_v2_140_224.h5 \
        --laion-mlp path/al/clip_autokeras_binary_nsfw \
        --out predictions/predictions_tf.csv
"""

import argparse
import csv
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent / "models"))
from cnn_gantman import GantManCNN  # noqa: E402
from clip_laion import LaionClipDetector  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--gantman-weights", required=True)
    parser.add_argument("--laion-mlp", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.test, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Evaluando {len(rows)} imagenes del test set...")

    models = [
        GantManCNN(args.gantman_weights),
        LaionClipDetector(args.laion_mlp),
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filepath", "class_name", "binary_label",
            "model_name", "nsfw_probability", "predicted_label", "latency_ms",
        ])

        for model in models:
            print(f"\n--- Modelo: {model.name} ---")
            for row in tqdm(rows):
                pred = model.predict(row["filepath"])
                writer.writerow([
                    row["filepath"], row["class_name"], row["binary_label"],
                    pred.model_name, pred.nsfw_probability,
                    pred.predicted_label, pred.latency_ms,
                ])

    print(f"\nListo. Resultados en {out_path}")


if __name__ == "__main__":
    main()
