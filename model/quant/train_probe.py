"""
Prueba de viabilidad del reentrenamiento (DES-1.3): memoria y tiempo de un paso de
entrenamiento del EVA02-Base a 448 px en esta GPU. Entradas aleatorias (solo mide recursos).

Uso (en el contenedor):
    python quant/train_probe.py
"""

import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageClassification


def probe(model, batch, ckpt, amp, steps=4):
    model.set_grad_checkpointing(ckpt)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for s in range(steps):
        x = torch.randn(batch, 3, 448, 448, device="cuda")
        y = torch.randint(0, 4, (batch,), device="cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = F.cross_entropy(model(x).float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        times.append(time.time() - t0)
    del opt
    return torch.cuda.max_memory_allocated() / 2 ** 30, sum(times[1:]) / (steps - 1)


def main():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, {props.total_memory / 2 ** 30:.1f} GiB")
    hf = AutoModelForImageClassification.from_pretrained(
        "Freepik/nsfw_image_detector", torch_dtype=torch.float32)
    model = hf.timm_model.cuda().train()
    print(f"{'batch':>5s} {'ckpt':>5s} {'amp':>5s} {'pico GiB':>9s} {'s/paso':>8s} {'s/imagen':>9s}")
    for ckpt, amp in ((True, True), (True, False), (False, True)):
        for batch in (1, 2, 4, 8):
            try:
                gib, t = probe(model, batch, ckpt, amp)
                print(f"{batch:5d} {str(ckpt):>5s} {str(amp):>5s} {gib:9.2f} {t:8.2f} {t / batch:9.2f}",
                      flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"{batch:5d} {str(ckpt):>5s} {str(amp):>5s}   sin memoria", flush=True)
                torch.cuda.empty_cache()
                break


if __name__ == "__main__":
    main()
