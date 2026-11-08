"""
Elimina fugas de datos entre los splits train/val/test usando el hash MD5 del
contenido. test.csv se conserva intacto (son las 504 imagenes del benchmark).

Reglas:
  1. Se quitan de train y val las imagenes cuyo hash esta en test.
  2. Se quitan de val las imagenes cuyo hash esta en train.
  3. Se deduplica dentro de cada split (se conserva la primera aparicion).
  4. Se excluyen de train y val los hashes con etiqueta binaria contradictoria
     (mismo contenido con clases NSFW y SFW).

NOTA: el script lee los CSV actuales; sobre los CSV ya deduplicados no detecta
los conflictos (ya se descarto una de las copias). Para regenerar desde cero,
restaurar train.csv/val.csv originales (git show c4226a4:split/train.csv).

Uso:
    python split/dedupe_splits.py --split-dir split --raw-root G:\\FPGate\\raw_data

Los CSV usan rutas del contenedor (/workspace/raw_data/...); --raw-root indica
donde esta raw_data en esta maquina. Escribe train.csv, val.csv (reescritos),
y hashes.csv (filepath, split, md5) para verificacion repetible.
"""

import argparse
import csv
import hashlib
from pathlib import Path

CONTAINER_PREFIX = "/workspace/raw_data/"


def md5_file(path: Path, block_size: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def read_split(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_split(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "class_name", "binary_label"])
        for r in rows:
            writer.writerow([r["filepath"], r["class_name"], r["binary_label"]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", default=Path("split"), type=Path)
    parser.add_argument("--raw-root", required=True, type=Path,
                        help="Ruta local de raw_data (equivale a /workspace/raw_data)")
    args = parser.parse_args()

    splits = {s: read_split(args.split_dir / f"{s}.csv") for s in ("train", "val", "test")}

    for name, rows in splits.items():
        for r in rows:
            rel = r["filepath"][len(CONTAINER_PREFIX):]
            r["md5"] = md5_file(args.raw_root / rel)
        print(f"hasheado {name}: {len(rows)}")

    test_hashes = {r["md5"] for r in splits["test"]}

    # Hashes con etiqueta binaria contradictoria entre/dentro de splits
    label_by_hash = {}
    conflict_hashes = set()
    for rows in splits.values():
        for r in rows:
            prev = label_by_hash.setdefault(r["md5"], r["binary_label"])
            if prev != r["binary_label"]:
                conflict_hashes.add(r["md5"])
    conflicts = len(conflict_hashes)

    seen_train = set()
    train = []
    t_test = t_dup = t_conf = 0
    for r in splits["train"]:
        if r["md5"] in conflict_hashes:
            t_conf += 1
        elif r["md5"] in test_hashes:
            t_test += 1
        elif r["md5"] in seen_train:
            t_dup += 1
        else:
            seen_train.add(r["md5"])
            train.append(r)

    seen_val = set()
    val = []
    v_test = v_train = v_dup = v_conf = 0
    for r in splits["val"]:
        if r["md5"] in conflict_hashes:
            v_conf += 1
        elif r["md5"] in test_hashes:
            v_test += 1
        elif r["md5"] in seen_train:
            v_train += 1
        elif r["md5"] in seen_val:
            v_dup += 1
        else:
            seen_val.add(r["md5"])
            val.append(r)

    cleaned = {"train": train, "val": val, "test": splits["test"]}

    print("\nEliminadas:")
    print(f"  train: por test={t_test}, duplicadas internas={t_dup}, etiqueta contradictoria={t_conf}")
    print(f"  val:   por test={v_test}, por train={v_train}, duplicadas internas={v_dup}, "
          f"etiqueta contradictoria={v_conf}")
    print(f"Hashes con etiqueta binaria contradictoria (excluidos): {conflicts}")

    hs = {n: {r["md5"] for r in cleaned[n]} for n in cleaned}
    assert not (hs["train"] & hs["val"]) and not (hs["train"] & hs["test"]) \
        and not (hs["val"] & hs["test"]), "quedan hashes compartidos"

    write_split(args.split_dir / "train.csv", cleaned["train"])
    write_split(args.split_dir / "val.csv", cleaned["val"])

    with open(args.split_dir / "hashes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "split", "md5"])
        for name in ("train", "val", "test"):
            for r in cleaned[name]:
                writer.writerow([r["filepath"], name, r["md5"]])

    for name, rows in cleaned.items():
        nsfw = sum(r["binary_label"] == "NSFW" for r in rows)
        print(f"{name}: {len(rows)} imagenes, NSFW={nsfw}, SFW={len(rows) - nsfw}")


if __name__ == "__main__":
    main()
