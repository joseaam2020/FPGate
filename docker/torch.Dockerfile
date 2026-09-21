
# Imagen para Freepik (ViT), LlavaGuard (VLM) y MobileViT (Hibrido):
# PyTorch + transformers + timm.

# Usa la imagen oficial de PyTorch con CUDA 12.1 ya instalado. Si la maquina
# no tiene GPU NVIDIA, el mismo contenedor corre en CPU sin cambios (torch
# detecta la ausencia de GPU automaticamente); solo sera mas lento.

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

COPY requirements/torch.txt /tmp/requirements.txt
# torch/torchvision ya vienen en la imagen base; evitar reinstalarlos
RUN grep -v -E "^torch(vision)?==" /tmp/requirements.txt > /tmp/requirements_filtered.txt \
    && pip install --no-cache-dir -r /tmp/requirements_filtered.txt

ENTRYPOINT ["python"]
