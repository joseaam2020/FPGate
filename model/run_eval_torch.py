"""
Corre Freepik (ViT) y LlavaGuard (VLM) sobre el test split congelado, y
escribe predictions_torch.csv.

Correr dentro de env_torch:
    python run_eval_torch.py --test splits/test.csv \
        --out predictions/predictions_torch.csv

Usar --skip-llavaguard si aun no se tiene acceso al modelo gateado, o para
correr una primera version del benchmark sin esperar la descarga.
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent / "models"))
from vit_freepik import FreepikEva02  # noqa: E402
from vlm_llavaguard import LlavaGuardVLM  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-llavaguard", action="store_true")
    args = parser.parse_args()

    with open(args.test, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Evaluando {len(rows)} imagenes del test")

    models = [
        FreepikEva02(),
    ]
    if not args.skip_llavaguard:
        models.append(LlavaGuardVLM())
    else:
        print("[INFO] Omitiendo LlavaGuard (--skip-llavaguard activo)")

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
