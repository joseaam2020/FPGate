# Imagen para GantMan (CNN) + LAION-AI (CLIP): TensorFlow/Keras + autokeras.

FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /workspace

COPY requirements/tf.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# El codigo se monta como volumen (ver docker-compose.yml), no se copia aqui,
# para poder editar los wrappers sin reconstruir la imagen cada vez.

ENTRYPOINT ["python"]
