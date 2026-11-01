import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons

from pathlib import Path
from PIL import Image, ImageFilter

# Prefijo de las rutas tal como quedaron escritas dentro del contenedor
CONTAINER_PREFIX = "/workspace/raw_data"

# Paleta (categorica, orden fijo) y tintas
COLOR_1 = "#2a78d6"   # azul
COLOR_2 = "#eb6834"   # naranja
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"

# Repos de Hugging Face de los que se descargan los pesos de cada modelo.
# CLIP-LAION = encoder CLIP ViT-L/14 (open_clip, pretrained="openai") + cabeza
# MLP local; el tamano total es la suma de ambos.
HF_REPOS = {
    "vit_freepik_eva02": "Freepik/nsfw_image_detector",
    "clip_laion_probed": "timm/vit_large_patch14_clip_224.openai",
    "vlm_llavaguard": "AIML-TUDA/LlavaGuard-v1.2-0.5B-OV-hf",
}


# ---------------------------------------------------------------- tamanos

def local_size_mb(path: Path) -> float:
    """Tamano de un archivo o de una carpeta completa (p. ej. SavedModel)."""
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def hf_weights_size_mb(repo_id: str) -> float:
    """Tamano del archivo de pesos de un repo de HF, sin descargarlo.

    Los repos suelen traer los mismos pesos en varios formatos (.safetensors,
    .bin, ...), asi que sumar todo sobreestimaria. Se toma el archivo
    .safetensors mas grande, o el .bin mas grande si no hay safetensors.
    """
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, files_metadata=True)
    files = [(s.rfilename, s.size or 0) for s in info.siblings or []]
    for ext in (".safetensors", ".bin"):
        candidates = [size for name, size in files if name.endswith(ext)]
        if candidates:
            return max(candidates) / 1e6
    raise RuntimeError(f"No se encontraron pesos en {repo_id}")


def model_sizes(args) -> dict:
    sizes = {}

    def try_size(label, fn):
        try:
            return fn()
        except Exception as ex:  # sin red, sin huggingface_hub, ruta mala...
            print(f"[WARN] No se pudo obtener el tamano de {label}: {ex}")
            return np.nan

    sizes["vit_freepik_eva02"] = try_size(
        "Freepik", lambda: hf_weights_size_mb(HF_REPOS["vit_freepik_eva02"]))

    clip = try_size("CLIP ViT-L/14", lambda: hf_weights_size_mb(HF_REPOS["clip_laion_probed"]))
    if args.laion_mlp:
        mlp = try_size("cabeza MLP de LAION", lambda: local_size_mb(Path(args.laion_mlp)))
    else:
        print("[WARN] Sin --laion-mlp: el tamano de CLIP-LAION solo incluye el encoder CLIP.")
        mlp = 0.0
    sizes["clip_laion_probed"] = clip + mlp

    sizes["vlm_llavaguard"] = try_size(
        "LlavaGuard", lambda: hf_weights_size_mb(HF_REPOS["vlm_llavaguard"]))

    sizes["cnn_gantman_mobilenetv2"] = (
        try_size("GantMan", lambda: local_size_mb(Path(args.gantman_weights)))
        if args.gantman_weights else np.nan)
    if args.gantman_weights is None:
        print("[WARN] Sin --gantman-weights: no se reporta el tamano de GantMan.")

    return sizes


# ---------------------------------------------------------------- metricas

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(correct=df["binary_label"] == df["predicted_label"])
    g = df.groupby("model_name")

    summary = pd.DataFrame({
        "n_imagenes": g.size(),
        "pct_correctas": g["correct"].mean() * 100,
        "pct_incorrectas": (1 - g["correct"].mean()) * 100,
        # Falsos positivos: SFW marcadas como NSFW; falsos negativos: al reves
        "falsos_positivos": g.apply(lambda d: ((d.binary_label == "SFW") & (d.predicted_label == "NSFW")).sum()),
        "falsos_negativos": g.apply(lambda d: ((d.binary_label == "NSFW") & (d.predicted_label == "SFW")).sum()),
    })

    lat = df.pivot_table(index="model_name", columns="predicted_label",
                         values="latency_ms", aggfunc=["mean", "median"])
    summary["latencia_prom_SFW_ms"] = lat[("mean", "SFW")]
    summary["latencia_prom_NSFW_ms"] = lat[("mean", "NSFW")]
    # La mediana se incluye porque la primera imagen de cada modelo carga la
    # GPU / el grafo y tarda mucho mas (warm-up), lo que sesga el promedio.
    summary["latencia_mediana_SFW_ms"] = lat[("median", "SFW")]
    summary["latencia_mediana_NSFW_ms"] = lat[("median", "NSFW")]
    return summary


# ---------------------------------------------------------------- graficas

def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2)
    ax.grid(axis="x", color=MUTED, alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_accuracy(summary: pd.DataFrame, out: Path):
    # Un pie por modelo, lado a lado, de mejor a peor
    s = summary.sort_values("pct_correctas", ascending=False)

    fig, axes = plt.subplots(1, len(s), figsize=(3.6 * len(s), 4.2), facecolor=SURFACE)
    axes = np.atleast_1d(axes)

    for ax, (model, r) in zip(axes, s.iterrows()):
        wedges, _, _ = ax.pie(
            [r["pct_correctas"], r["pct_incorrectas"]],
            colors=[COLOR_1, COLOR_2],
            startangle=90, counterclock=False,
            # Porcentajes afuera del circulo: la rebanada de errores suele
            # ser demasiado delgada para que el numero quepa adentro
            autopct="%1.1f%%", pctdistance=1.22,
            wedgeprops={"edgecolor": SURFACE, "linewidth": 2},
            textprops={"color": INK, "fontsize": 11},
        )
        ax.set_title(f"{model}\n(n={r['n_imagenes']})", color=INK, fontsize=10, pad=24)
        ax.set_aspect("equal")

    fig.legend(wedges, ["Correctas", "Incorrectas"], loc="lower center",
               ncol=2, frameon=False)
    fig.suptitle("Clasificaciones correctas vs. incorrectas por modelo", color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=150)
    return fig


def plot_latency(summary: pd.DataFrame, out: Path):
    s = summary.sort_values("latencia_prom_NSFW_ms")
    y = np.arange(len(s))
    h = 0.35

    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.9 * len(s)), facecolor=SURFACE)
    style_axes(ax)
    bars_sfw = ax.barh(y - h / 2, s["latencia_prom_SFW_ms"], height=h, color=COLOR_1,
                       edgecolor=SURFACE, linewidth=2, label="Predicha SFW")
    bars_nsfw = ax.barh(y + h / 2, s["latencia_prom_NSFW_ms"], height=h, color=COLOR_2,
                        edgecolor=SURFACE, linewidth=2, label="Predicha NSFW")
    ax.bar_label(bars_sfw, fmt="%.0f ms", padding=4, color=INK_2, fontsize=9)
    ax.bar_label(bars_nsfw, fmt="%.0f ms", padding=4, color=INK_2, fontsize=9)

    ax.set_yticks(y, s.index, color=INK)
    ax.set_xlabel("Latencia promedio (ms)", color=INK_2)
    ax.set_title("Latencia promedio por tipo de clasificacion", color=INK, loc="left")
    ax.margins(x=0.15)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    return fig


def plot_sizes(sizes: dict, out: Path):
    s = pd.Series(sizes).dropna().sort_values()
    if s.empty:
        print("[WARN] Ningun tamano disponible; se omite la grafica de pesos.")
        return
    y = np.arange(len(s))

    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.7 * len(s)), facecolor=SURFACE)
    style_axes(ax)
    bars = ax.barh(y, s.values, height=0.55, color=COLOR_1, edgecolor=SURFACE, linewidth=2)
    ax.bar_label(bars, fmt="%.0f MB", padding=4, color=INK_2, fontsize=9)
    ax.set_yticks(y, s.index, color=INK)
    ax.set_xlabel("Peso en disco (MB)", color=INK_2)
    ax.set_title("Peso de cada modelo", color=INK, loc="left")
    ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    return fig


# ---------------------------------------------------------------- galeria

def local_path(container_path: str, images_root: Path) -> Path:
    rel = container_path.replace(CONTAINER_PREFIX, "", 1).lstrip("/")
    return images_root / rel


class ErrorGallery:
    """Ventana interactiva para explorar las imagenes mal clasificadas.

    Controles:
        - Modelo: elige de que modelo ver los errores
        - Tipo de error: todos / falsos positivos / falsos negativos
        - Difuminar: activa/desactiva el blur de las imagenes
        - Anterior / Siguiente (o flechas del teclado): cambia de pagina
        - Clic en una imagen: la abre en grande con todos sus detalles
    """

    ROWS, COLS = 3, 4
    # Filtro -> etiqueta REAL de los errores que se muestran
    FILTERS = {
        "Todos": None,
        "Falsos positivos\n(SFW -> NSFW)": "SFW",
        "Falsos negativos\n(NSFW -> SFW)": "NSFW",
    }

    def __init__(self, errors: pd.DataFrame, images_root, blur: bool):
        # Primero los errores con mayor confianza (los mas "graves")
        self.errors = (errors.assign(conf=(errors["nsfw_probability"] - 0.5).abs())
                       .sort_values("conf", ascending=False))
        self.images_root = images_root
        self.models = sorted(self.errors["model_name"].unique())
        self.model = self.models[0]
        self.filter = "Todos"
        self.blur = blur
        self.page = 0
        self.cache = {}

        self.fig = plt.figure(figsize=(14, 8.5), facecolor=SURFACE)
        try:
            self.fig.canvas.manager.set_window_title("Galeria de errores")
        except AttributeError:
            pass

        # --- panel de controles (izquierda)
        ax = self.fig.add_axes((0.01, 0.58, 0.2, 0.3), facecolor=SURFACE)
        ax.set_title("Modelo", color=INK, loc="left", fontsize=10)
        self.radio_model = RadioButtons(ax, self.models, active=0)
        self.radio_model.on_clicked(self._on_model)

        ax = self.fig.add_axes((0.01, 0.3, 0.2, 0.22), facecolor=SURFACE)
        ax.set_title("Tipo de error", color=INK, loc="left", fontsize=10)
        self.radio_filter = RadioButtons(ax, list(self.FILTERS), active=0)
        self.radio_filter.on_clicked(self._on_filter)

        ax = self.fig.add_axes((0.01, 0.2, 0.2, 0.06), facecolor=SURFACE)
        self.check_blur = CheckButtons(ax, ["Difuminar"], [blur])
        self.check_blur.on_clicked(self._on_blur)

        self.btn_prev = Button(self.fig.add_axes((0.01, 0.08, 0.095, 0.06)), "< Anterior")
        self.btn_next = Button(self.fig.add_axes((0.115, 0.08, 0.095, 0.06)), "Siguiente >")
        self.btn_prev.on_clicked(lambda _: self._go(-1))
        self.btn_next.on_clicked(lambda _: self._go(+1))

        # --- cuadricula de miniaturas (derecha)
        gs = self.fig.add_gridspec(self.ROWS, self.COLS, left=0.25, right=0.99,
                                   top=0.88, bottom=0.03, hspace=0.5, wspace=0.08)
        self.cells = [self.fig.add_subplot(gs[i, j])
                      for i in range(self.ROWS) for j in range(self.COLS)]
        self.cell_rows: list = [None] * len(self.cells)

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.render()

    # --- datos

    def current(self) -> pd.DataFrame:
        d = self.errors[self.errors["model_name"] == self.model]
        true_label = self.FILTERS[self.filter]
        if true_label is not None:
            d = d[d["binary_label"] == true_label]
        return d

    def load_image(self, filepath: str, max_side: int):
        if self.images_root is None:
            return None
        key = (filepath, max_side, self.blur)
        if key not in self.cache:
            try:
                img = Image.open(local_path(filepath, self.images_root)).convert("RGB")
                img.thumbnail((max_side, max_side))
                if self.blur:
                    img = img.filter(ImageFilter.GaussianBlur(radius=max_side / 30))
                self.cache[key] = img
            except Exception:
                self.cache[key] = None
        return self.cache[key]

    # --- dibujo

    def render(self):
        d = self.current()
        per_page = len(self.cells)
        pages = max(1, int(np.ceil(len(d) / per_page)))
        self.page = min(max(self.page, 0), pages - 1)
        chunk = d.iloc[self.page * per_page:(self.page + 1) * per_page]

        for i, ax in enumerate(self.cells):
            ax.clear()
            ax.axis("off")
            if i >= len(chunk):
                self.cell_rows[i] = None
                continue
            r = chunk.iloc[i]
            self.cell_rows[i] = r
            img = self.load_image(r["filepath"], 300)
            if img is not None:
                ax.imshow(img)
            else:
                msg = "sin --images-root" if self.images_root is None else "imagen no\nencontrada"
                ax.text(0.5, 0.5, msg, ha="center", va="center", color=MUTED,
                        transform=ax.transAxes)
            ax.set_title(f"real: {r['binary_label']} ({r['class_name']})\n"
                         f"pred: {r['predicted_label']}  p={r['nsfw_probability']:.2f}",
                         fontsize=8, color=INK)

        self.fig.suptitle(
            f"{self.model}  |  {self.filter.splitlines()[0]}: {len(d)} errores  |  "
            f"pagina {self.page + 1}/{pages}\n"
            "clic en una imagen para ampliarla  -  flechas <- -> para cambiar de pagina",
            color=INK, fontsize=11)
        self.fig.canvas.draw_idle()

    def show_detail(self, r: pd.Series):
        """Abre una ventana aparte con la imagen en grande y sus detalles."""
        others = self.errors[(self.errors["filepath"] == r["filepath"])
                             & (self.errors["model_name"] != r["model_name"])]["model_name"]

        fig = plt.figure(figsize=(9, 8), facecolor=SURFACE)
        ax = fig.add_axes((0.02, 0.22, 0.96, 0.74))
        ax.axis("off")
        img = self.load_image(r["filepath"], 1000)
        if img is not None:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "imagen no disponible", ha="center", va="center",
                    color=MUTED, transform=ax.transAxes)
        fig.text(0.03, 0.03,
                 f"Modelo: {r['model_name']}\n"
                 f"Etiqueta real: {r['binary_label']} (clase original: {r['class_name']})\n"
                 f"Prediccion: {r['predicted_label']}   P(NSFW) = {r['nsfw_probability']:.4f}   "
                 f"latencia = {r['latency_ms']:.0f} ms\n"
                 f"Tambien la fallaron: {', '.join(others) if len(others) else 'ningun otro modelo'}\n"
                 f"{r['filepath']}",
                 fontsize=9, color=INK, family="monospace", va="bottom")
        fig.show()

    # --- eventos

    def _go(self, step: int):
        self.page += step
        self.render()

    def _on_model(self, label):
        self.model, self.page = label, 0
        self.render()

    def _on_filter(self, label):
        self.filter, self.page = label, 0
        self.render()

    def _on_blur(self, _):
        self.blur = not self.blur
        self.render()

    def _on_key(self, event):
        if event.key == "right":
            self._go(+1)
        elif event.key == "left":
            self._go(-1)

    def _on_click(self, event):
        if event.button != 1 or event.inaxes not in self.cells:
            return
        r = self.cell_rows[self.cells.index(event.inaxes)]
        if r is not None:
            self.show_detail(r)


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf", default="tf_results.csv")
    parser.add_argument("--torch", default="torch_results.csv")
    parser.add_argument("--out", default="analysis", type=Path)
    parser.add_argument("--images-root", type=Path,
                        help="Carpeta local de raw_data (reemplaza /workspace/raw_data)")
    parser.add_argument("--gantman-weights", help="Archivo/carpeta del modelo de GantMan")
    parser.add_argument("--laion-mlp", help="Carpeta de la cabeza MLP de LAION")
    parser.add_argument("--blur", action="store_true",
                        help="Abre la galeria con las imagenes difuminadas")
    parser.add_argument("--no-show", action="store_true",
                        help="Solo guarda los archivos, sin abrir las ventanas")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.concat([pd.read_csv(args.tf), pd.read_csv(args.torch)], ignore_index=True)

    summary = summarize(df)
    sizes = model_sizes(args)
    summary["peso_MB"] = pd.Series(sizes)

    summary.round(2).to_csv(args.out / "resumen.csv")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("\n" + summary.round(2).to_string())

    plot_accuracy(summary, args.out / "correctas_vs_incorrectas.png")
    plot_latency(summary, args.out / "latencia.png")
    plot_sizes(sizes, args.out / "peso_modelos.png")

    errors = df[df["binary_label"] != df["predicted_label"]]
    errors.to_csv(args.out / "errores.csv", index=False)

    print(f"\nListo. Resultados en {args.out.resolve()}")

    if args.no_show:
        return
    if args.images_root is None:
        print("[WARN] Sin --images-root: la galeria muestra solo los datos, sin imagenes.")
    if errors.empty:
        print("[INFO] Ningun modelo tuvo errores; no hay galeria.")
    else:
        gallery = ErrorGallery(errors, args.images_root, args.blur)  # noqa: F841 (mantener referencia viva)
    plt.show()


if __name__ == "__main__":
    main()
