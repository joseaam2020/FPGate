#!/usr/bin/env python3
"""Compares the proxy's image preprocessing with Pillow, byte for byte.

The reference is what the model sees at training/evaluation time:
  Image.open(f).convert("RGB") -> Resize(shorter side, BICUBIC) -> CenterCrop
(timm's eval transform for EVA02), or a plain squash resize.

usage: test_prep.py PREP_TOOL
"""
import io
import os
import random
import struct
import subprocess
import sys
import tempfile
import zlib

from PIL import Image, ImageDraw, ImageFilter

TOOL = sys.argv[1]
TMP = tempfile.mkdtemp()
failed = 0


def check(name, cond, extra=""):
    global failed
    print(("PASS " if cond else "FAIL ") + name + (f"  {extra}" if extra and not cond else ""))
    if not cond:
        failed = 1


def photo(w, h, seed):
    """Noise + blur + shapes: smooth areas and sharp edges, like a real picture."""
    rnd = random.Random(seed)
    im = Image.frombytes("RGB", (w, h), rnd.randbytes(w * h * 3)).filter(ImageFilter.GaussianBlur(3))
    d = ImageDraw.Draw(im)
    for _ in range(12):
        x0, y0 = rnd.randrange(w), rnd.randrange(h)
        d.rectangle([x0, y0, x0 + rnd.randrange(1, w // 2 + 2), y0 + rnd.randrange(1, h // 2 + 2)],
                    fill=tuple(rnd.randrange(256) for _ in range(3)))
    return im


def ref(path, size, mode):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if mode == "squash":
        return im.resize((size, size), Image.BICUBIC).tobytes()
    short, long_ = (w, h) if w <= h else (h, w)
    new_long = int(size * long_ / short)
    nw, nh = (size, new_long) if w <= h else (new_long, size)
    if (w <= h and w == size) or (h <= w and h == size):
        pass                                   # torchvision returns the image untouched
    else:
        im = im.resize((nw, nh), Image.BICUBIC)
    nw, nh = im.size
    left, top = int(round((nw - size) / 2.0)), int(round((nh - size) / 2.0))
    return im.crop((left, top, left + size, top + size)).tobytes()


def run_tool(path, size, mode, dct=0, max_pixels=None):
    out = os.path.join(TMP, "out.rgb")
    if os.path.exists(out):
        os.unlink(out)
    env = dict(os.environ)
    if max_pixels:
        env["PREP_MAX_PIXELS"] = str(max_pixels)
    r = subprocess.run([TOOL, path, str(size), mode, str(dct), out], capture_output=True, env=env)
    data = open(out, "rb").read() if r.returncode == 0 else None
    return r.returncode, data, r.stderr.decode(errors="replace")


def save(im, name, **kw):
    path = os.path.join(TMP, name)
    im.save(path, **kw)
    return path


def compare(label, path, size, mode):
    rc, data, err = run_tool(path, size, mode)
    want = ref(path, size, mode)
    if rc != 0:
        check(label, False, f"tool exit {rc}: {err.strip()[:200]}")
        return
    if data == want:
        check(label, True)
    else:
        diff = max(abs(a - b) for a, b in zip(data, want))
        check(label, False, f"max diff {diff}, differing bytes {sum(a != b for a, b in zip(data, want))}")


def png_adam7(im):
    """Hand-written interlaced PNG (Pillow cannot write them)."""
    w, h = im.size
    px = im.load()
    passes = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
    raw = bytearray()
    for x0, y0, dx, dy in passes:
        xs, ys = range(x0, w, dx), range(y0, h, dy)
        if not len(xs) or not len(ys):
            continue
        for y in ys:
            raw.append(0)
            for x in xs:
                raw += bytes(px[x, y])

    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 1))
            + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b""))


def main():
    # --- JPEG: every geometry, both modes -------------------------------------
    jpeg_cases = [
        ("landscape 1200x800", photo(1200, 800, 1), dict(quality=85)),
        ("portrait 600x900", photo(600, 900, 2), dict(quality=85)),
        ("exact 448x448", photo(448, 448, 3), dict(quality=90)),
        ("upscale 300x200", photo(300, 200, 4), dict(quality=90)),
        ("tiny 33x77", photo(33, 77, 5), dict(quality=75)),
        ("progressive 800x600", photo(800, 600, 6), dict(quality=85, progressive=True)),
        ("4:4:4 640x480", photo(640, 480, 7), dict(quality=90, subsampling=0)),
        ("4:2:2 500x500", photo(500, 500, 8), dict(quality=80, subsampling=1)),
        ("extreme aspect 2000x300", photo(2000, 300, 9), dict(quality=85)),
        ("large 3000x2000", photo(3000, 2000, 10), dict(quality=88)),
    ]
    for label, im, kw in jpeg_cases:
        path = save(im, "t.jpg", **kw)
        for size, mode in ((448, "crop"), (224, "squash")):
            compare(f"jpeg {label} -> {size} {mode}", path, size, mode)
    compare("jpeg grayscale 700x500 -> 448 crop", save(photo(700, 500, 11).convert("L"), "g.jpg", quality=85), 448, "crop")
    compare("jpeg 1x1 -> 224 squash", save(Image.new("RGB", (1, 1), (10, 200, 30)), "p.jpg"), 224, "squash")

    # --- PNG ------------------------------------------------------------------
    rnd = random.Random(99)
    rgba = photo(640, 480, 12).convert("RGBA")
    rgba.putalpha(Image.frombytes("L", (640, 480), rnd.randbytes(640 * 480)))
    png_cases = [
        ("RGB 640x480", photo(640, 480, 13)),
        ("RGBA with random alpha", rgba),
        ("L 500x300", photo(500, 300, 14).convert("L")),
        ("LA", photo(400, 400, 15).convert("LA")),
        ("palette", photo(300, 450, 16).quantize(64)),
        ("1-bit", photo(320, 240, 17).convert("1")),
    ]
    for label, im in png_cases:
        path = save(im, "t.png")
        for size, mode in ((448, "crop"), (224, "squash")):
            compare(f"png {label} -> {size} {mode}", path, size, mode)
    p = os.path.join(TMP, "adam7.png")
    open(p, "wb").write(png_adam7(photo(333, 211, 18)))
    compare("png interlaced (Adam7) 333x211 -> 448 crop", p, 448, "crop")
    compare("png interlaced (Adam7) 333x211 -> 224 squash", p, 224, "squash")

    # --- DCT scaling: not bit-exact, but must stay close ----------------------
    big = save(photo(3000, 2000, 10), "big.jpg", quality=88)
    rc, scaled, err = run_tool(big, 448, "crop", dct=1)
    exact = ref(big, 448, "crop")
    mad = sum(abs(a - b) for a, b in zip(scaled or b"", exact)) / len(exact) if scaled else 99
    check("jpeg DCT scaling stays within 3 grey levels on average", rc == 0 and mad < 3, f"mean abs diff {mad:.2f}")

    # --- damaged / hostile input ------------------------------------------------
    jpg = open(save(photo(800, 600, 20), "c.jpg", quality=85), "rb").read()
    png = open(save(photo(400, 300, 21), "c.png"), "rb").read()

    def tool_on(data, name="x.bin", **kw):
        path = os.path.join(TMP, name)
        open(path, "wb").write(data)
        return run_tool(path, 448, "crop", **kw)[0]

    check("truncated JPEG is reported corrupt", tool_on(jpg[: len(jpg) // 2]) == 3)
    check("truncated PNG is reported corrupt", tool_on(png[: len(png) // 2]) == 3)
    check("JPEG magic followed by garbage is corrupt", tool_on(b"\xff\xd8\xff" + os.urandom(2000)) == 3)
    bad = bytearray(png)
    bad[len(bad) // 2] ^= 0xFF
    check("PNG with a flipped byte (bad CRC) is corrupt", tool_on(bytes(bad)) == 3)
    check("empty input is not an image", tool_on(b"") == 6)
    check("GIF is recognised but not decodable", tool_on(b"GIF89a" + bytes(64)) == 6)

    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body))

    bomb = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 30000, 30000, 8, 2, 0, 0, 0))
    check("PNG claiming 900 megapixels is refused before decoding", tool_on(bomb, max_pixels=16000000) == 5)
    cmyk = save(Image.new("CMYK", (64, 64), (10, 20, 30, 40)), "cmyk.jpg")
    check("CMYK JPEG is unsupported", run_tool(cmyk, 448, "crop")[0] == 4)

    sys.exit(failed)


if __name__ == "__main__":
    main()
