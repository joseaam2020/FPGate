"""
Compara las rondas de poda (r1-r8) y r7 contra los modelos iniciales, leyendo SOLO archivos
de resultados (no corre ningun modelo). Solo usa la biblioteca estandar.

Fuentes: quant/results/{prune_r*,std500_r*,hold_*,test_*}.json y *_logits.csv, analysis/resumen.csv,
quant/results/profile.json (opcional, lo escribe profile_model.py) y quant/release/*/manifest.json.

Salidas en quant/results/comparison/: comparison.md, comparison.html (con graficas SVG) y un CSV por tabla.
Las muestras (val[:500], holdout de 1000, test de 504) son distintas y se reportan en tablas separadas.

Uso (en el contenedor o con cualquier Python 3):
    python quant/compare.py
"""

import csv
import glob
import html
import json
import math
import re
from collections import Counter
from pathlib import Path

RES = Path("quant/results")
OUT = RES / "comparison"
HD, DIM, PATCH_W, HEAD_W = 64, 768, 768 * 3 * 14 * 14, 4 * 768
ORIG_DIMS = [(12, 2048)] * 12
NOSTAT = "—"

# criterio de aceptacion de docs/e1_referencia.md, secciones 6 y 7 (test, 504 imagenes)
LIM = {"INT8": dict(acc=476, fn=11, fp=18, flips=10), "INT8 podado": dict(acc=472, fn=12, fp=20, flips=20)}


def jload(p):
    return json.load(open(p, encoding="utf-8")) if Path(p).exists() else None


def int8_weights(dims):
    """Bytes de pesos INT8 de las GEMM (patch_embed + q,k,v,proj + fc1g,fc1x,fc2 + head)."""
    tot = PATCH_W + HEAD_W
    for h, n in dims:
        tot += 4 * DIM * h * HD + 3 * DIM * n
    return tot


def pct(x):
    return f"{x * 100:.1f}%"


def cls_of(path):
    m = re.search(r"/raw_data/([^/]+)/", path.replace("\\", "/"))
    return m.group(1) if m else "?"


def verdict_counts(logits_csv, col="frozen"):
    """FN y FP por clase desde un CSV de logits (formato de eval_int.py o de eval_test.py)."""
    fn, fp = Counter(), Counter()
    with open(logits_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if col + "_l0" in r:
                l = [float(r[f"{col}_l{j}"]) for j in range(4)]
                m = max(l)
                e = [math.exp(v - m) for v in l]
                nsfw = sum(e[1:]) / sum(e) >= 0.5
            else:
                nsfw = float(r["nsfw_prob"]) >= 0.5
            c = r.get("class_name") or cls_of(r["filepath"])
            pos = r["binary_label"] == "NSFW"
            if pos and not nsfw:
                fn[c] += 1
            if not pos and nsfw:
                fp[c] += 1
    return fn, fp


def fmt_cls(c):
    return ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])) or "0"


# ---------------------------------------------------------------- tabla A: rondas en val[:500]
def rounds_table(profile):
    rows = []
    teacher = None
    for pf in sorted(glob.glob(str(RES / "prune_r*_*.json")), key=lambda p: int(re.search(r"prune_r(\d+)", p).group(1))):
        short = "r" + re.search(r"prune_r(\d+)", pf).group(1)
        pj = jload(pf)
        sj = jload(RES / f"std500_{short}.json")
        dims = [tuple(d) for d in pj["dims"]]
        heads = sorted({h for h, _ in dims})
        neurons = sorted({n for _, n in dims})
        r = dict(ronda=short, cabezas="/".join(map(str, heads)), neuronas="/".join(map(str, neurons)),
                 ops=pj["ops_fraction"], peso_int8_mb=int8_weights(dims) / 2**20,
                 tag=pj["args"]["tag"], entrenadas=pj["args"]["train_images"])
        if sj:
            s = sj["summary"]
            teacher = teacher or s["teacher"]
            fz = s["frozen"]
            r.update(acc=fz["acc"], fn=fz["fn"], fp=fz["fp"], cambios=fz["flips_vs_teacher"],
                     err=fz["logit_err_vs_teacher"])
        rows.append(r)
    return rows, teacher


# ----------------------------------------------------- tabla B: holdout de 1000 imagenes
def holdout_table():
    specs = [("hold_unpruned", "INT8 sin podar"), ("hold_r5", "r5 INT8"), ("hold_r7", "r7 INT8")]
    rows, teacher_done = [], False
    for tag, name in specs:
        j = jload(RES / f"{tag}.json")
        if not j:
            continue
        s = j["summary"]
        if "teacher" in s and not teacher_done:
            t = s["teacher"]
            fn, fp = verdict_counts(RES / f"{tag}_logits.csv", "teacher")
            rows.append(dict(modelo="Original flotante", ops=1.0, acc=t["acc"], fn=t["fn"], fp=t["fp"],
                             cambios=0, fn_cls=fmt_cls(fn), fp_cls=fmt_cls(fp)))
            teacher_done = True
        fz = s["frozen"]
        fn, fp = verdict_counts(RES / f"{tag}_logits.csv", "frozen")
        ops = 1.0
        m = re.search(r"hold_(r\d+)", tag)
        if m:
            pj = next(iter(glob.glob(str(RES / f"prune_{m.group(1)}_*.json"))), None)
            ops = jload(pj)["ops_fraction"] if pj else None
        rows.append(dict(modelo=name, ops=ops, acc=fz["acc"], fn=fz["fn"], fp=fz["fp"],
                         cambios=fz.get("flips_vs_teacher", fz["flips_vs_float"]),
                         fn_cls=fmt_cls(fn), fp_cls=fmt_cls(fp)))
    rows.sort(key=lambda r: r["modelo"] != "Original flotante")   # el original va primero
    return rows


# ------------------------------------------------------------ tabla C: test de 504 imagenes
def test_table(profile):
    rows = []
    for r in csv.DictReader(open("analysis/resumen.csv", encoding="utf-8")):
        n = int(r["n_imagenes"])
        rows.append(dict(modelo=r["model_name"], n=n, acc=float(r["pct_correctas"]) / 100,
                         fn=int(r["falsos_negativos"]), fp=int(r["falsos_positivos"]),
                         cambios=NOSTAT, fn_cls=NOSTAT, fp_cls=NOSTAT, ops=NOSTAT,
                         lat_ms=float(r["latencia_mediana_NSFW_ms"]) if r["latencia_mediana_NSFW_ms"] else None,
                         peso_mb=float(r["peso_MB"]) if r["peso_MB"] else None, origen="benchmark",
                         cumple=NOSTAT))
    for tag, name, key in [("test_final", "ViT INT8 sin podar", "INT8"), ("test_r7", "ViT r7 INT8 podado", "INT8 podado")]:
        j = jload(RES / f"{tag}.json")
        if not j:
            continue
        i = j["int8"]
        fn, fp = verdict_counts(RES / f"{tag}_logits.csv")
        lim = LIM[key]
        ok = i["correct"] >= lim["acc"] and i["fn"] <= lim["fn"] and i["fp"] <= lim["fp"] and i["flips_vs_ref"] <= lim["flips"]
        ops = 1.0
        if tag == "test_r7":
            ops = jload(RES / "prune_r7_heads33_neur40.json")["ops_fraction"]
        rows.append(dict(modelo=name, n=j["n_images"], acc=i["acc"], fn=i["fn"], fp=i["fp"], cambios=i["flips_vs_ref"],
                         fn_cls=fmt_cls(fn), fp_cls=fmt_cls(fp), ops=ops, lat_ms=None, origen="quant/eval_test.py",
                         peso_mb=None, cumple=("sí" if ok else "no (excepción §8)" if tag == "test_r7" else "no")))
    # peso INT8 exportado de r7 y latencias de profile.json
    for p in glob.glob("quant/release/*/manifest.json"):
        m = jload(p)
        size = sum(Path(p).parent.joinpath(n).stat().st_size for n in m["files"]) / 2**20
        for r in rows:
            if r["modelo"] == "ViT r7 INT8 podado" and "r7" in m["name"]:
                r["peso_mb"] = size
    return rows


# ------------------------------------------------ tabla D: perfil (MACs, tamano, latencia)
def profile_table(profile):
    if not profile:
        return []
    rows = []
    base = profile["models"]["unpruned"]
    for name, d in profile["models"].items():
        g, c = d.get("gpu_fp32"), d.get("cpu_fp32")
        rows.append(dict(modelo="original" if name == "unpruned" else name, gmacs=d["gmacs"], ops=d["ops_vs_original"],
                         params_m=d["params_m"], fp32_mb=d["fp32_mb"], int8_mb=d["int8_gemm_mb"],
                         gpu_ms=g["median_ms"] if g else None, cpu_ms=c["median_ms"] if c else None,
                         gpu_red=(1 - g["median_ms"] / base["gpu_fp32"]["median_ms"]) if g else None,
                         cpu_red=(1 - c["median_ms"] / base["cpu_fp32"]["median_ms"]) if c else None))
    return rows


# -------------------------------------------------------------------------------- salida
def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f(x, fmt="{:.1f}"):
    return NOSTAT if x is None or x == NOSTAT else (fmt.format(x) if not isinstance(x, str) else x)


def svg_scatter(points, title, xlabel, ylabel, w=560, h=320):
    """points: [(x, y, label, highlight)] -> SVG inline (ejes simples)."""
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    x0, x1, y0, y1 = min(xs) - 0.03, max(xs) + 0.03, 0, max(ys) * 1.2 + 1
    L, R, T, B = 48, 14, 30, 40
    px = lambda x: L + (x - x0) / (x1 - x0) * (w - L - R)
    py = lambda y: h - B - (y - y0) / (y1 - y0) * (h - T - B)
    s = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}" style="max-width:100%;height:auto">',
         f'<text x="{L}" y="18" font-size="13" font-weight="600" fill="currentColor">{html.escape(title)}</text>',
         f'<line x1="{L}" y1="{h - B}" x2="{w - R}" y2="{h - B}" stroke="currentColor" opacity=".4"/>',
         f'<line x1="{L}" y1="{T}" x2="{L}" y2="{h - B}" stroke="currentColor" opacity=".4"/>',
         f'<text x="{w / 2}" y="{h - 6}" font-size="11" text-anchor="middle" fill="currentColor">{html.escape(xlabel)}</text>',
         f'<text x="12" y="{h / 2}" font-size="11" text-anchor="middle" transform="rotate(-90 12 {h / 2})" fill="currentColor">{html.escape(ylabel)}</text>']
    for t in range(0, int(y1) + 1, max(1, int(y1) // 5)):
        s.append(f'<text x="{L - 6}" y="{py(t) + 4}" font-size="10" text-anchor="end" fill="currentColor">{t}</text>')
    for t in [round(x0 + i * (x1 - x0) / 5, 2) for i in range(6)]:
        s.append(f'<text x="{px(t)}" y="{h - B + 14}" font-size="10" text-anchor="middle" fill="currentColor">{t:.2f}</text>')
    for x, y, lab, hi in points:
        col = "#d1495b" if hi else "#2e6f9e"
        s.append(f'<circle cx="{px(x)}" cy="{py(y)}" r="{6 if hi else 4.5}" fill="{col}"><title>{html.escape(lab)}: ops {x:.2f}, {y}</title></circle>')
        s.append(f'<text x="{px(x) + 7}" y="{py(y) - 6}" font-size="10" fill="currentColor">{html.escape(lab)}</text>')
    s.append("</svg>")
    return "".join(s)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    profile = jload(RES / "profile.json")
    A, teacher = rounds_table(profile)
    B = holdout_table()
    C = test_table(profile)
    D = profile_table(profile)

    orig_mb = int8_weights(ORIG_DIMS) / 2**20
    md = ["# Comparación de resultados E1 (generado por quant/compare.py)", "",
          "Solo lee archivos de resultados. Las tres muestras son distintas y no se mezclan.", ""]

    md += ["## A. Rondas de poda — val[:500], modelo entero congelado",
           f"Original flotante (maestro) en la misma muestra: acc {pct(teacher['acc'])}, FN {teacher['fn']}, FP {teacher['fp']}. "
           f"Peso INT8 del original (solo GEMM): {orig_mb:.1f} MB. Cabezas y neuronas = valores por bloque (neuronas distintas por el bloque 1 protegido).", "",
           md_table(["Ronda", "Cabezas", "Neuronas", "Ops (% orig.)", "Peso INT8 (MB)", "Acc", "FN", "FP", "Cambios vs. maestro", "Err. logits", "Imgs. entren."],
                    [[r["ronda"], r["cabezas"], r["neuronas"], pct(r["ops"]), f(r["peso_int8_mb"]),
                      f(r.get("acc"), "{:.1%}"), r.get("fn", NOSTAT), r.get("fp", NOSTAT), r.get("cambios", NOSTAT),
                      f(r.get("err"), "{:.3f}"), r["entrenadas"]] for r in A]), ""]

    md += ["## B. Holdout — 1000 imágenes nunca usadas, modelo entero congelado", "",
           md_table(["Modelo", "Ops", "Acc", "FN", "FP", "Cambios vs. original", "FN por clase", "FP por clase"],
                    [[r["modelo"], pct(r["ops"]) if r["ops"] is not None else NOSTAT, f(r["acc"], "{:.1%}"), r["fn"], r["fp"],
                      r["cambios"], r["fn_cls"], r["fp_cls"]] for r in B]), ""]

    md += ["## C. Test — 504 imágenes (LlavaGuard: 159)", "",
           md_table(["Modelo", "N", "Acc", "FN", "FP", "Cambios vs. benchmark", "FN por clase", "FP por clase", "Ops", "Peso (MB)", "Latencia mediana (ms)", "Cumple criterio E1"],
                    [[r["modelo"], r["n"], f(r["acc"], "{:.2%}"), r["fn"], r["fp"], r["cambios"], r["fn_cls"], r["fp_cls"],
                      pct(r["ops"]) if isinstance(r["ops"], float) else NOSTAT, f(r["peso_mb"]), f(r["lat_ms"], "{:.0f}"), r["cumple"]]
                     for r in C]), "",
           "Peso: benchmark = peso del modelo en disco (HF); r7 = paquete exportado (INT8 + tablas + parámetros del HPS). "
           "Latencia: del benchmark (GPU, fp16); la de r7 no es comparable, ver tabla D. "
           "Un guion indica dato no medido.", ""]
    if D:
        pc = lambda x: NOSTAT if x is None else f"{x:.0%}"
        md += ["## D. Perfil de cómputo y tamaño (original y rondas)",
               f"Latencia: float_ref FP32 por imagen, mediana ({profile['gpu']}, CPU {profile['cpu_threads']} hilos). "
               "No es la latencia del acelerador; mide cuánto baja el cómputo al podar. "
               "Ops aquí incluye patch_embed y cabeza (r7: 65.1 %; las tablas A y C cuentan solo los bloques: 64.9 %). "
               "Peso INT8 = solo matrices GEMM.", "",
               md_table(["Modelo", "GMACs", "Ops", "Parámetros (M)", "FP32 (MB)", "INT8 GEMM (MB)", "Lat. GPU (ms)", "Δ GPU", "Lat. CPU (ms)", "Δ CPU"],
                        [[r["modelo"], f(r["gmacs"]), pct(r["ops"]), f(r["params_m"]), f(r["fp32_mb"]), f(r["int8_mb"]),
                          f(r["gpu_ms"], "{:.0f}"), pc(r["gpu_red"]), f(r["cpu_ms"], "{:.0f}"), pc(r["cpu_red"])] for r in D]), ""]
    (OUT / "comparison.md").write_text("\n".join(md), encoding="utf-8")

    for name, rows in (("rounds_val500", A), ("holdout", B), ("test", C), ("profile", D)):
        keys = sorted({k for r in rows for k in r})
        with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    pts_fn = [(r["ops"], r["fn"], r["ronda"], r["ronda"] == "r7") for r in A if "fn" in r]
    pts_fp = [(r["ops"], r["fp"], r["ronda"], r["ronda"] == "r7") for r in A if "fp" in r]
    pts_fl = [(r["ops"], r["cambios"], r["ronda"], r["ronda"] == "r7") for r in A if "cambios" in r]
    figs = "".join(f"<div>{svg_scatter(p, t, 'fracción de operaciones', 'imágenes')}</div>" for p, t in
                   ((pts_fn, f"FN en val[:500] (maestro: {teacher['fn']})"), (pts_fp, f"FP en val[:500] (maestro: {teacher['fp']})"),
                    (pts_fl, "Cambios de veredicto vs. maestro (val[:500])")))
    body = re.sub(r"^# .*\n", "", "\n".join(md))
    sections = ""
    for block in re.split(r"\n(?=## )", body):
        lines = block.strip().split("\n")
        if not lines or not lines[0].startswith("##"):
            continue
        sections += f"<h2>{html.escape(lines[0][3:])}</h2>"
        tbl = [l for l in lines if l.startswith("|")]
        for l in lines[1:]:
            if not l.startswith("|") and l.strip():
                sections += f"<p>{html.escape(l)}</p>"
        if tbl:
            hdr = [c.strip() for c in tbl[0].strip("|").split("|")]
            sections += "<div class=scroll><table><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in hdr) + "</tr></thead><tbody>"
            for l in tbl[2:]:
                cells = [c.strip() for c in l.strip("|").split("|")]
                hi = ' class="hi"' if cells[0] in ("r7", "r7 INT8", "ViT r7 INT8 podado") else ""
                sections += f"<tr{hi}>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>"
            sections += "</tbody></table></div>"
    page = f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Comparación E1</title><style>
:root{{--bg:#fff;--fg:#1c2530;--mut:#5b6673;--line:#d9dee4;--hi:#fff1f2}}
@media(prefers-color-scheme:dark){{:root{{--bg:#12161b;--fg:#e4e8ec;--mut:#9aa5b1;--line:#2b333c;--hi:#3a1d22}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:16px}}
h1{{font-size:22px}}h2{{font-size:17px;margin-top:28px}}p{{color:var(--mut)}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;font-size:13px;width:100%}}
th,td{{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;white-space:nowrap}}tr.hi td{{background:var(--hi)}}
.figs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}}</style></head><body>
<h1>Comparación de resultados E1</h1><p>Generado por quant/compare.py a partir de archivos de resultados. r7 resaltado.</p>
<h2>Gráficas — val[:500]</h2><div class=figs>{figs}</div>{sections}</body></html>"""
    (OUT / "comparison.html").write_text(page, encoding="utf-8")
    print("\n".join(md))
    print(f"\nescrito en {OUT}/ (comparison.md, comparison.html, *.csv)")


if __name__ == "__main__":
    main()
