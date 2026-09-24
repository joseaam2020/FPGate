BINARY_LABEL = {
    "porn": "NSFW",
    "hentai": "NSFW",
    "sexy": "NSFW",
    "neutral": "SFW",
    "drawings": "SFW",
}

CLASSES = list(BINARY_LABEL.keys())


def to_binary(class_name: str) -> str:
    """Convierte un nombre de clase original (nombre de carpeta) a 'NSFW' o 'SFW'."""
    class_name = class_name.strip().lower()
    if class_name not in BINARY_LABEL:
        raise ValueError(
            f"Clase desconocida: '{class_name}'. Se esperaba una de {CLASSES}"
        )
    return BINARY_LABEL[class_name]