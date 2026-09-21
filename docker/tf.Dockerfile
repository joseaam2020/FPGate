# Imagen para GantMan (CNN) + LAION-AI (CLIP): TensorFlow/Keras + autokeras.
# Utilizando CPU

FROM tensorflow/tensorflow:2.15.0

WORKDIR /workspace

COPY requirements/tf.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# El codigo se monta como volumen (ver docker-compose.yml), no se copia aqui,
# para poder editar los wrappers sin reconstruir la imagen cada vez.

ENTRYPOINT ["python"]
