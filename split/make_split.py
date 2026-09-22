"""
Construye los splits FIJOS de train, val y test a partir de la carpeta raw_data
generada por deepghs/nsfw_detect.

Uso:
    python data/make_splits.py --raw-data G:\\nsfw_data_scraper\\raw_data --out splits/

Estratificación: se estratifica por la ETIQUETA BINARIA (NSFW/SFW), no por la
clase original de 5 vías. Esto asegura que train/val/test mantengan la misma
proporción NSFW/SFW (evita que, por azar, el test set quede con muy pocas
imágenes NSFW). Dentro de cada split, se preserva además la distribución
relativa de las 5 clases originales (ver --stratify-5class).
"""

import argparse
import csv
import hashlib
from pathlib import Path
from sklearn.model_selection import train_test_split


SEED = 42 

BINARY_LABEL = {
    "porn": "NSFW",
    "hentai": "NSFW",
    "sexy": "NSFW",
    "neutral": "SFW",
    "drawings": "SFW",
}

BINARY_LABEL_INT = {
    "porn": 1,
    "hentai": 1,
    "sexy": 1,
    "neutral": 0,
    "drawings": 0,
}

CLASSES = list(BINARY_LABEL.keys())


def list_images(raw_data_dir: Path):
    """Recorre raw_data/<clase>/... y regresa lista de (path, clase)."""
    rows = []
    for class_name in CLASSES:
        class_dir = raw_data_dir / class_name
        if not class_dir.exists():
            print(f"[WARN] No existe la carpeta: {class_dir} — se omite.")
            continue
        # recursivo, por si el scraper anidó subcarpetas (p. ej. por subreddit)
        for path in class_dir.rglob("*"):
            if path.is_file():
                rows.append((str(path.resolve()), class_name))
    return rows


def file_hash(path: str, block_size: int = 65536) -> str:
    """Hash rápido del contenido del archivo, usado solo para detectar
    duplicados EXACTOS dentro del propio dataset (no contra GantMan/LAION,
    ver limitación de data leakage documentada en la bitácora, Paso 2)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def to_binary(class_name: str) -> str:
    """Convierte un nombre de clase original (nombre de carpeta) a 'NSFW' o 'SFW'."""
    class_name = class_name.strip().lower()
    if class_name not in BINARY_LABEL:
        raise ValueError(
            f"Clase desconocida: '{class_name}'. Se esperaba una de {CLASSES}"
        )
    return BINARY_LABEL[class_name]


def to_binary_int(class_name: str) -> int:
    """Igual que to_binary, pero retorna 1 (NSFW) / 0 (SFW) para cálculo de métricas."""
    class_name = class_name.strip().lower()
    if class_name not in BINARY_LABEL_INT:
        raise ValueError(
            f"Clase desconocida: '{class_name}'. Se esperaba una de {CLASSES}"
        )
    return BINARY_LABEL_INT[class_name]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data", required=True, type=Path,
                         help="Ruta a la carpeta raw_data del scraper")
    parser.add_argument("--out", default=Path("splits"), type=Path,
                         help="Carpeta donde se escriben train.csv/val.csv/test.csv")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--drop-exact-duplicates", action="store_true",
                         help="Elimina duplicados exactos (mismo hash) antes de splitear")
    args = parser.parse_args()

    assert abs(args.train_frac + args.val_frac + args.test_frac - 1.0) < 1e-6, \
        "Las fracciones train/val/test deben sumar 1.0"

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Escaneando {args.raw_data} ...")
    rows = list_images(args.raw_data)
    # Orden estable: rglob() no garantiza el mismo orden entre corridas ni
    # entre filesystems, y train_test_split depende del orden de entrada
    # incluso con random_state fijo. Sin este sort, el split cambiaría entre
    # ejecuciones pese a SEED=42.
    rows.sort()
    print(f"Total de archivos encontrados: {len(rows)}")

    if args.drop_exact_duplicates:
        print("Calculando hashes para eliminar duplicados exactos (puede tardar)...")
        seen = set()
        deduped = []
        for path, class_name in rows:
            h = file_hash(path)
            if h in seen:
                continue
            seen.add(h)
            deduped.append((path, class_name))
        print(f"Duplicados exactos eliminados: {len(rows) - len(deduped)}")
        rows = deduped

    paths = [r[0] for r in rows]
    classes = [r[1] for r in rows]
    binary_labels = [to_binary(c) for c in classes]

    # Split 1: separar train vs (val+test), estratificado por etiqueta binaria
    train_paths, rest_paths, train_classes, rest_classes = train_test_split(
        paths, classes,
        train_size=args.train_frac,
        stratify=binary_labels,
        random_state=SEED,
    )

    # Split 2: separar val vs test dentro del resto, misma proporción relativa
    rest_binary = [to_binary(c) for c in rest_classes]
    val_size_rel = args.val_frac / (args.val_frac + args.test_frac)
    val_paths, test_paths, val_classes, test_classes = train_test_split(
        rest_paths, rest_classes,
        train_size=val_size_rel,
        stratify=rest_binary,
        random_state=SEED,
    )

    splits = {
        "train": list(zip(train_paths, train_classes)),
        "val": list(zip(val_paths, val_classes)),
        "test": list(zip(test_paths, test_classes)),
    }

    for split_name, split_rows in splits.items():
        out_path = args.out / f"{split_name}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filepath", "class_name", "binary_label"])
            for path, class_name in split_rows:
                writer.writerow([path, class_name, to_binary(class_name)])
        print(f"Escrito: {out_path} ({len(split_rows)} imágenes)")

    # Resumen de conteo por split x clase
    print("\n--- Resumen ---")
    for split_name, split_rows in splits.items():
        counts = {}
        for _, class_name in split_rows:
            counts[class_name] = counts.get(class_name, 0) + 1
        nsfw = sum(v for k, v in counts.items() if to_binary(k) == "NSFW")
        sfw = sum(v for k, v in counts.items() if to_binary(k) == "SFW")
        print(f"{split_name}: {counts}  -> NSFW={nsfw}, SFW={sfw}")


if __name__ == "__main__":
    main()
