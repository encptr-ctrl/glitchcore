#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# глитчкор из картинки
# нужны pillow и numpy (pip install pillow numpy)
#
# python glitchcore.py -> окно выбора файла
# python glitchcore.py картинка.jpg -> сразу делает
# python glitchcore.py картинка.jpg -o папка -> в папку
# python glitchcore.py --no-gui -> пути в консоли
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageFilter, ImageOps

EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

# ink - цвет линий шума, a1 и a2 акценты
PRESETS = {
    "red":    dict(ink=(240, 240, 245), a1=(232, 18, 40),  a2=(120, 70, 255)),
    "violet": dict(ink=(236, 226, 255), a1=(255, 60, 150), a2=(170, 60, 255)),
    "ice":    dict(ink=(235, 245, 255), a1=(40, 200, 255), a2=(90, 80, 255)),
    "acid":   dict(ink=(240, 255, 235), a1=(60, 255, 120), a2=(255, 40, 120)),
}


# --- пути ---

def clean_path(p):
    return os.path.expanduser(p.strip().strip('"').strip("'"))


def ask_file(use_gui):
    if use_gui:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            p = filedialog.askopenfilename(
                title="Выбери картинку для глитча",
                filetypes=[("Изображения", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
                           ("Все файлы", "*.*")])
            root.destroy()
            if not p:
                sys.exit("Файл не выбран.")
            return p
        except Exception:
            pass  # нету окошка, спросим в консоли
    p = input("Путь к картинке (можно перетащить файл сюда): ")
    return clean_path(p)


def ask_folder(default, use_gui):
    if use_gui:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            p = filedialog.askdirectory(title="Куда сохранить результат? (Отмена = рядом с исходником)",
                                        initialdir=default)
            root.destroy()
            return p or default
        except Exception:
            pass
    p = input(f"Папка для сохранения [Enter = {default}]: ").strip()
    return clean_path(p) if p else default


# --- загрузка ---

def parse_hex(s):
    s = s.lstrip("#")
    if len(s) != 6:
        raise argparse.ArgumentTypeError("цвет нужен в формате #rrggbb")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def _load_img(path, long_side):
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    w, h = im.size
    k = long_side / max(w, h)
    return im.resize((max(16, round(w * k)), max(16, round(h * k))), Image.LANCZOS)


def _load_gray(path, long_side):
    return np.asarray(_load_img(path, long_side).convert("L")).astype(np.float32) / 255.0


def parse_rect(s):
    try:
        v = [float(x) for x in s.split(",")]
        assert len(v) == 4 and 0 <= v[0] < v[2] <= 1 and 0 <= v[1] < v[3] <= 1
    except Exception:
        raise argparse.ArgumentTypeError("надо 4 числа от 0 до 1: x0,y0,x1,y1 (например 0.2,0,1,1)")
    return tuple(v)


def load_tones(path, long_side, bg_mode="auto", tol=20.0, rect=None, mask_path=None):
    """для цветных рисунков, тона + маска фона"""
    img = _load_img(path, long_side)
    rgb = np.asarray(img)
    g = np.asarray(img.convert("L")).astype(np.float32) / 255.0
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    tones = np.clip((g - lo) / (hi - lo + 1e-6), 0, 1)
    tones = np.clip((tones - 0.5) * 1.5 + 0.5, 0, 1)
    if g.mean() < 0.25:
        tones = tones ** 0.7  # тёмные картинки чуть светлее, а то перс тонет
    return tones, find_bg(rgb, g, bg_mode, tol, rect, mask_path)


# --- поиск фона ---

def border_colors(rgb):
    """основные цвета по краю кадра (до 4 штук)"""
    H, W = rgb.shape[:2]
    t = max(2, int(min(H, W) * 0.02))
    strip = np.concatenate([rgb[:t].reshape(-1, 3), rgb[-t:].reshape(-1, 3),
                            rgb[:, :t].reshape(-1, 3), rgb[:, -t:].reshape(-1, 3)]).astype(np.float32)
    q = (strip // 32).astype(np.int32)
    key = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    vals, cnt = np.unique(key, return_counts=True)
    cols = []
    for i in np.argsort(-cnt)[:4]:
        if cnt[i] < 0.1 * len(strip) and cols:
            break
        cols.append(strip[key == vals[i]].mean(axis=0))
    return np.array(cols)


def _morph(mask, k, op):
    """открытие/закрытие маски через PIL, шоб без scipy работало"""
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    lo, hi = ImageFilter.MinFilter(k), ImageFilter.MaxFilter(k)
    a, b = (lo, hi) if op == "open" else (hi, lo)
    return np.asarray(im.filter(a).filter(b)) > 127


def bg_color(rgb, tol):
    """фон = то что по цвету похоже на края кадра и связано с краем. работает и на тёмном и на светлом"""
    H, W = rgb.shape[:2]
    soft = np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(max(1.0, min(H, W) / 500))))
    cols = border_colors(soft)
    f = soft.astype(np.float32)
    d = np.min([np.sqrt(((f - c) ** 2).sum(-1)) for c in cols], axis=0)
    cand = d < tol
    cand = _morph(cand, 3, "open")  # шоб не протекало через тонкие щели
    return border_connected(cand)


def bg_grabcut(rgb, rect, seed_bg):
    """для сложных случаев когда фон по цвету сливаеться с персом. нужен opencv"""
    import cv2
    H, W = rgb.shape[:2]
    k = min(1.0, 540.0 / max(H, W))
    w, h = max(16, int(W * k)), max(16, int(H * k))
    sm = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    m = np.full((h, w), cv2.GC_PR_FGD, np.uint8)
    t = max(2, int(min(h, w) * 0.02))
    sb = cv2.resize(seed_bg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    sb = np.asarray(Image.fromarray(sb.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(7))) > 127  # сужаем шоб край не задеть
    if sb.sum() > 0.01 * h * w:
        m[sb] = cv2.GC_BGD
    else:
        m[:t] = m[-t:] = cv2.GC_BGD
        m[:, :t] = m[:, -t:] = cv2.GC_BGD
    if rect:
        x0, y0, x1, y1 = rect
        ins = np.zeros((h, w), bool)
        ins[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = True
        m[~ins] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(cv2.cvtColor(sm, cv2.COLOR_RGB2BGR), m, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
    fg = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    fg = cv2.resize(fg, (W, H), interpolation=cv2.INTER_LINEAR) > 127
    return ~fg


def clean_fg(fg):
    """убираем мелкие островки перса в фоне и сглаживаем край"""
    try:
        from scipy import ndimage
        lab, n = ndimage.label(fg)
        if n:
            sizes = ndimage.sum(fg, lab, range(1, n + 1))
            small = np.isin(lab, 1 + np.where(sizes < 0.0015 * fg.size)[0])
            fg = fg & ~small
    except ImportError:
        pass
    im = Image.fromarray(fg.astype(np.uint8) * 255).filter(ImageFilter.MedianFilter(5))
    return np.asarray(im) > 127


def find_bg(rgb, g, mode, tol, rect, mask_path):
    H, W = g.shape
    if mask_path:  # руками нарисованная маска: белое = перс
        m = Image.open(mask_path).convert("L").resize((W, H), Image.BILINEAR)
        return ~(np.asarray(m) > 127)
    if mode == "light":
        bg = border_connected(g > 0.88)
    elif mode == "grabcut":
        try:
            bg = bg_grabcut(rgb, rect, bg_color(rgb, tol))
        except ImportError:
            sys.exit("для --bg grabcut нужен opencv: pip install opencv-python")
    else:
        bg = bg_color(rgb, tol)
    if rect and mode != "grabcut":
        x0, y0, x1, y1 = rect
        outside = np.ones((H, W), bool)
        outside[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = False
        bg = bg | outside
    fg = clean_fg(~bg)
    fg = np.asarray(Image.fromarray(fg.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(3))) > 127  # -1px шоб светлой каймы не было
    return ~fg


def border_connected(mask):
    """фон = светлое что касаеться края"""
    try:
        from scipy import ndimage
        lab, _ = ndimage.label(mask)
        edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        return np.isin(lab, edge[edge > 0])
    except ImportError:
        pass
    H, W = mask.shape  # если нет scipy
    flat = mask.ravel().tolist()
    seen = bytearray(H * W)
    stack = []
    for x in range(W):
        for i in (x, (H - 1) * W + x):
            if flat[i] and not seen[i]:
                seen[i] = 1
                stack.append(i)
    for y in range(H):
        for i in (y * W, y * W + W - 1):
            if flat[i] and not seen[i]:
                seen[i] = 1
                stack.append(i)
    while stack:
        i = stack.pop()
        x = i % W
        for j in (i - W, i + W, i - 1 if x > 0 else -1, i + 1 if x < W - 1 else -1):
            if 0 <= j < H * W and flat[j] and not seen[j]:
                seen[j] = 1
                stack.append(j)
    return np.frombuffer(bytes(seen), dtype=np.uint8).reshape(H, W) > 0


def load_ink(path, long_side, invert):
    g = _load_gray(path, long_side)
    if g.mean() < 0.3 or g.mean() > 0.9:  # почти вся картинка тёмная/светлая -> выравниваем, иначе штрихов нет
        hist, edges = np.histogram(g, 256, (0, 1))
        g = np.interp(g, edges[:-1], hist.cumsum() / g.size).astype(np.float32)
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    g = np.clip((g - lo) / (hi - lo + 1e-6), 0, 1)
    g = np.clip((g - 0.5) * 2.4 + 0.5, 0, 1)
    ink = 1.0 - g  # чернила это 1
    return 1.0 - ink if invert else ink


# --- глитчи ---

def blend(canvas, mask, color):
    m = np.clip(mask, 0, 1)[..., None]
    return canvas * (1 - m) + np.array(color, np.float32) / 255.0 * m


def run_map(rng, n, maxrun):
    """повторяем строки рандомно (потёки)"""
    runs = rng.integers(1, max(2, maxrun), size=n)
    cs = np.cumsum(runs)
    k = min(int(np.searchsorted(cs, n)) + 1, n)
    idx = np.linspace(0, n - 1, k).astype(int)
    return np.repeat(idx, runs[:k])[:n]


def scale_of(a):
    return max(a.shape) / 1080.0


def stroke_layer(ink, rng, n, smin, smax):
    """кусочки арта увеличеные и повернутые, тока контуры"""
    H, W = ink.shape
    m = min(H, W)
    base = Image.fromarray((ink * 255).astype(np.uint8))
    layer = np.zeros((H, W), np.float32)
    for _ in range(n):
        cs = int(rng.integers(max(16, int(.11 * m)), max(17, int(.39 * m))))
        cs = min(cs, W, H)
        x = int(rng.integers(0, W - cs + 1))
        y = int(rng.integers(0, H - cs + 1))
        crop = base.crop((x, y, x + cs, y + cs))
        side = max(4, int(cs * rng.uniform(smin, smax)))
        crop = crop.resize((side, side), Image.BILINEAR)
        crop = crop.rotate(float(rng.uniform(-90, 90)), expand=True, resample=Image.BILINEAR)
        a = np.asarray(crop).astype(np.float32) / 255.0
        gy, gx = np.gradient(a)
        a = np.clip(np.hypot(gx, gy) * 9.0, 0, 1)
        ph, pw = a.shape
        px = int(rng.integers(-pw // 3, max(-pw // 3 + 1, W - pw // 2)))
        py = int(rng.integers(-ph // 3, max(-ph // 3 + 1, H - ph // 2)))
        x0, y0 = max(px, 0), max(py, 0)
        x1, y1 = min(px + pw, W), min(py + ph, H)
        if x1 <= x0 or y1 <= y0:
            continue
        sub = a[y0 - py:y1 - py, x0 - px:x1 - px]
        layer[y0:y1, x0:x1] = np.maximum(layer[y0:y1, x0:x1], sub)
    return layer


def vstretch(layer, rng, count, maxrun=40):
    H, W = layer.shape
    f = scale_of(layer)
    out = layer.copy()
    for _ in range(count):
        w = int(rng.integers(max(2, int(.02 * W)), max(3, int(.15 * W))))
        x0 = int(rng.integers(0, max(1, W - w)))
        src = run_map(rng, H, int(maxrun * f))
        out[:, x0:x0 + w] = layer[src][:, x0:x0 + w]
    return out


def hstretch(layer, rng, count, maxrun=30):
    H, W = layer.shape
    f = scale_of(layer)
    out = layer.copy()
    for _ in range(count):
        hh = int(rng.integers(max(2, int(.01 * H)), max(3, int(.08 * H))))
        y0 = int(rng.integers(0, max(1, H - hh)))
        src = run_map(rng, W, int(maxrun * f))
        out[y0:y0 + hh, :] = layer[y0:y0 + hh][:, src]
    return out


def mosaic(layer, rng, n):
    H, W = layer.shape
    f = scale_of(layer)
    for _ in range(n):
        cell = max(4, int(rng.choice([16, 24, 32, 40, 48]) * f))
        bw = min(int(rng.integers(6, 18)) * cell, (W // cell) * cell)
        bh = min(int(rng.integers(5, 16)) * cell, (H // cell) * cell)
        if bw < 2 * cell or bh < 2 * cell:
            continue
        x = int(rng.integers(0, W - bw + 1))
        y = int(rng.integers(0, H - bh + 1))
        reg = layer[y:y + bh, x:x + bw]
        small = reg.reshape(bh // cell, cell, bw // cell, cell).mean(axis=(1, 3))
        small = (small > rng.uniform(0.08, 0.3)).astype(np.float32)
        if rng.random() < 0.15:
            small = 1 - small
        layer[y:y + bh, x:x + bw] = np.repeat(np.repeat(small, cell, 0), cell, 1)
    return layer


def pixel_sort(layer, rng, n):
    H, W = layer.shape
    f = scale_of(layer)
    for _ in range(n):
        ww = max(1, int(rng.integers(4, 36) * f))
        x = int(rng.integers(0, max(1, W - ww)))
        y0 = int(rng.integers(0, max(1, int(.8 * H))))
        ln = int(rng.integers(int(.14 * H) + 1, int(.55 * H) + 2))
        y1 = min(H, y0 + ln)
        for xx in range(x, min(W, x + ww)):
            col = np.sort(layer[y0:y1, xx])
            layer[y0:y1, xx] = col if rng.random() < 0.5 else col[::-1]
    return layer


def diag(layer, rng, n):
    H, W = layer.shape
    for _ in range(n):
        bw = min(W, int(rng.integers(int(.18 * W) + 2, int(.44 * W) + 3)))
        bh = min(H, int(rng.integers(int(.18 * H) + 2, int(.48 * H) + 3)))
        x = int(rng.integers(0, W - bw + 1))
        y = int(rng.integers(0, H - bh + 1))
        reg = layer[y:y + bh, x:x + bw].copy()
        k = rng.uniform(0.3, 0.9) * rng.choice([-1, 1])
        for r in range(bh):
            reg[r] = np.roll(reg[r], int(r * k), axis=0)
        layer[y:y + bh, x:x + bw] = reg
    return layer


def light_glitch(ink, rng):
    """чуть двигаем фигуру по горизонтали"""
    H, W = ink.shape[:2]
    f = scale_of(ink)
    a = ink.copy()
    for _ in range(18):
        y = int(rng.integers(0, max(1, H - 4)))
        hh = max(2, int(rng.integers(2, 10) * f))
        off = int(rng.integers(-int(.065 * W) - 1, int(.065 * W) + 2))
        a[y:y + hh] = np.roll(a[y:y + hh], off, axis=1)
    return a


# --- сборка ---

def compose_tones(canvas, tones, bg, rng, colors, f, outline=False):
    """--tones: фон шум, перс как был. обводку вернуть --outline"""
    H, W = tones.shape
    fm = (~bg).astype(np.float32)  # маска без расширения а то белая обводка лезет

    st = light_glitch(np.dstack([tones, fm]), rng)

    # пиксельные квадраты на перса
    for _ in range(5):
        cell = max(6, int(rng.integers(14, 34) * f))
        bw = min(int(rng.integers(4, 10)) * cell, (W // cell) * cell)
        bh = min(int(rng.integers(3, 8)) * cell, (H // cell) * cell)
        if bw < 2 * cell or bh < 2 * cell:
            continue
        x = int(rng.integers(0, W - bw + 1))
        y = int(rng.integers(0, H - bh + 1))
        reg = st[y:y + bh, x:x + bw]
        small = reg.reshape(bh // cell, cell, bw // cell, cell, 2).mean(axis=(1, 3))
        st[y:y + bh, x:x + bw] = np.repeat(np.repeat(small, cell, 0), cell, 1)
    tn, fm = st[..., 0], st[..., 1]

    if outline:
        dx, dy = max(1, int(9 * f)), max(1, int(2 * f))
        canvas = blend(canvas, np.roll(fm, (-dy, -dx), (0, 1)) * 0.9, colors["a2"])
        canvas = blend(canvas, np.roll(fm, (dy, dx), (0, 1)) * 0.9, colors["a1"])

    fig = tn[..., None] * (np.array(colors["ink"], np.float32) / 255.0)
    for _ in range(4):  # цветные полоски на перса
        bw = int(rng.integers(int(.08 * W), int(.3 * W) + 2))
        bh = int(rng.integers(int(.01 * H) + 1, int(.06 * H) + 2))
        x = int(rng.integers(0, max(1, W - bw)))
        y = int(rng.integers(0, max(1, H - bh)))
        col = np.array(colors["a1"] if rng.random() < 0.6 else colors["a2"], np.float32) / 255.0
        fig[y:y + bh, x:x + bw] = tn[y:y + bh, x:x + bw, None] * col * 1.3
    fig = np.clip(fig, 0, 1)
    m = fm[..., None]
    return canvas * (1 - m) + fig * m


def render(ink, seed, colors, density=1.0, voids=0.44, tones=None, bg=None, outline=False):
    rng = np.random.default_rng(seed)
    H, W = ink.shape
    f = scale_of(ink)
    m = min(H, W)
    area_k = (W * H) / float(m * m) * density

    def build(n, mos, vs, hs, ps, dg):
        L = stroke_layer(ink, rng, max(1, int(n * area_k)), 1.4, 4.5)
        L = mosaic(L, rng, int(mos * area_k))
        L = vstretch(L, rng, vs)
        L = hstretch(L, rng, hs)
        L = pixel_sort(L, rng, int(ps * area_k))
        L = diag(L, rng, dg)
        return L

    Lw = build(130, 8, 12, 14, 30, 4)
    Lr = build(20, 0, 4, 3, 8, 2)
    Lv = build(16, 0, 4, 3, 6, 2)

    speck = (rng.random((H, W)) > 0.965).astype(np.float32)
    speck = vstretch(speck, rng, 20, 25)

    canvas = np.zeros((H, W, 3), np.float32)
    canvas = blend(canvas, Lw * 0.85, colors["ink"])
    canvas = blend(canvas, Lv * 0.75, colors["a2"])
    canvas = blend(canvas, Lr * 0.85, colors["a1"])

    # чёрные дырки шоб чёрного больше было
    gh = max(3, round(14 * H / m))
    gw = max(3, round(14 * W / m))
    low = rng.random((gh, gw)).astype(np.float32)
    low = np.asarray(Image.fromarray((low * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)).astype(np.float32) / 255.0
    low = vstretch(low, rng, 8, 12)
    keep = np.clip((low - voids) * 30.0, 0, 1)
    canvas *= (0.06 + 0.94 * keep)[..., None]
    canvas = blend(canvas, speck * 0.45, colors["ink"])

    if tones is not None:
        canvas = compose_tones(canvas, tones, bg, rng, colors, f, outline)
        canvas[::3] *= 0.95
        canvas += rng.normal(0, 0.02, canvas.shape).astype(np.float32)
        return Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8))

    fig = light_glitch(ink, rng)

    # чёрная рамка вокруг фигуры шоб видно было
    k = max(3, int(9 * f))
    k += (k % 2 == 0)
    hal = Image.fromarray((np.clip(fig * 1.3, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(k))
    hal = np.asarray(hal.filter(ImageFilter.GaussianBlur(1.2 * f))).astype(np.float32) / 255.0
    canvas = blend(canvas, np.clip(hal * 1.1, 0, 1) * 0.9, (0, 0, 0))

    # цветные тени и белая линия
    dx, dy = max(1, int(9 * f)), max(1, int(2 * f))
    canvas = blend(canvas, np.roll(fig, (-dy, -dx), (0, 1)) * 0.9, colors["a2"])
    canvas = blend(canvas, np.roll(fig, (dy, dx), (0, 1)) * 0.9, colors["a1"])
    canvas = blend(canvas, fig * 0.97, (255, 255, 255))

    canvas[::3] *= 0.95
    canvas += rng.normal(0, 0.02, canvas.shape).astype(np.float32)
    canvas = np.clip(canvas, 0, 1)
    return Image.fromarray((canvas * 255).astype(np.uint8))


# --- стиль stretch (растяжка пикселей) ---

# top/mid/bot - цвета сверху/по середине/снизу кадра
STRETCH_PALS = {
    "red":    dict(top=(22, 28, 150), mid=(105, 70, 235), bot=(232, 6, 42)),
    "violet": dict(top=(70, 20, 170), mid=(190, 60, 230), bot=(255, 70, 160)),
    "ice":    dict(top=(10, 40, 140), mid=(40, 150, 235), bot=(120, 90, 255)),
    "acid":   dict(top=(10, 110, 80), mid=(60, 210, 120), bot=(255, 40, 120)),
}


def drips(a, rng, n, f, wmax=40):
    """тонкие рваные потёки, у каждой колонки свой ряд и длина"""
    H, W = a.shape[:2]
    for _ in range(n):
        w = int(rng.integers(2, max(4, wmax * f)))
        x0 = int(rng.integers(0, max(1, W - w)))
        y0 = int(rng.integers(0, H))
        ys = np.clip(y0 + np.cumsum(rng.integers(-5, 6, w)), 0, H - 1)
        L0 = int(rng.integers(int(.08 * H) + 3, int(.9 * H) + 4))
        down = rng.random() < 0.75
        k = int(rng.integers(1, 14))
        for i in range(w):
            L = max(8, L0 + int(rng.integers(-L0 // 3, L0 // 3 + 1)))
            step = np.arange(L) // k
            if down:
                y1 = min(H, ys[i] + L)
                src = np.minimum(ys[i] + step[:y1 - ys[i]], H - 1)
                a[ys[i]:y1, x0 + i] = a[src, x0 + i]
            else:
                y1 = max(0, ys[i] - L)
                src = np.maximum(ys[i] - step[:ys[i] - y1], 0)
                a[y1:ys[i], x0 + i] = a[src[::-1], x0 + i]
    return a


def diag_streaks(a, rng, n):
    """тянем по горизонтали и двигаем строки -> длинные диагональные потёки"""
    H, W = a.shape[:2]
    for _ in range(n):
        bw = int(rng.integers(int(.45 * W), W))
        bh = int(rng.integers(int(.3 * H), int(.8 * H)))
        x = int(rng.integers(0, W - bw + 1))
        y = int(rng.integers(0, H - bh + 1))
        reg = a[y:y + bh, x:x + bw]
        reg = reg[:, run_map(rng, bw, 40)]
        k = rng.uniform(0.45, 1.0)
        shift = -(np.arange(bh) * k).astype(int)
        cols = (np.arange(bw)[None, :] + shift[:, None]) % bw
        reg = reg[np.arange(bh)[:, None], cols]
        if rng.random() < 0.5:  # частая штриховка сверху
            yy, xx = np.mgrid[0:bh, 0:bw]
            per = int(rng.integers(3, 6))
            line = ((yy * 1.0 + xx * 0.7) % per) < per * 0.75
            reg = np.where(line[..., None], reg, 1.0)
        a[y:y + bh, x:x + bw] = reg
    return a


def hslices(a, rng, n, maxoff):
    H = a.shape[0]
    for _ in range(n):
        y = int(rng.integers(0, H - 4))
        hh = int(rng.integers(2, 14))
        a[y:y + hh] = np.roll(a[y:y + hh], int(rng.integers(-maxoff, maxoff)), axis=1)
    return a


def mosaic_rgb(a, rng, n, f):
    H, W = a.shape[:2]
    for _ in range(n):
        cell = int(rng.choice([6, 10, 14, 20]) * f)
        bw = min(int(rng.integers(6, 20)) * cell, (W // cell) * cell)
        bh = min(int(rng.integers(5, 16)) * cell, (H // cell) * cell)
        if bw < 2 * cell or bh < 2 * cell:
            continue
        x = int(rng.integers(0, W - bw + 1))
        y = int(rng.integers(0, H - bh + 1))
        reg = a[y:y + bh, x:x + bw]
        small = reg.reshape(bh // cell, cell, bw // cell, cell, 3).mean(axis=(1, 3))
        a[y:y + bh, x:x + bw] = np.repeat(np.repeat(small, cell, 0), cell, 1)
    return a


def smooth_noise(rng, H, W, cells):
    ch = max(2, cells)
    cw = max(2, int(cells * W / H))
    low = rng.random((ch, cw)).astype(np.float32)
    return np.asarray(Image.fromarray((low * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)).astype(np.float32) / 255.0


def render_stretch(img, seed, pal, amount=0.7):
    rng = np.random.default_rng(seed)
    H, W = img.shape[:2]
    f = max(1, round(max(H, W) / 1080))

    work = img.copy()
    work = diag_streaks(work, rng, int(3 * amount))
    work = drips(work, rng, int(110 * amount), f, 36)
    work = mosaic_rgb(work, rng, int(3 * amount), f)
    work = hslices(work, rng, int(18 * amount), max(2, int(.06 * W)))
    work = drips(work, rng, int(60 * amount), f, 12)

    # плотность краски = насколько пиксель отличается от фона (а не просто тёмный), так силуэт видно даже у светлых волос
    bgc = np.median(border_colors(np.asarray(Image.fromarray((img * 255).astype(np.uint8)))), axis=0) / 255.0

    gl = img @ np.array([0.299, 0.587, 0.114], np.float32)
    glo, ghi = np.percentile(gl, 2), np.percentile(gl, 98)
    light_bg = float(bgc @ np.array([0.299, 0.587, 0.114])) > 0.8

    def density(im):
        if light_bg:  # светлый фон: силуэт = чем отличаешься от фона
            dist = np.sqrt(((im - bgc) ** 2).sum(-1))
            dk = np.clip(1.0 - im @ np.array([0.299, 0.587, 0.114], np.float32), 0, 1)
            return np.clip(dist * 0.55 + dk * 0.6, 0, 1)
        # картинка на весь кадр (фона нет): плотность = тёмное относительно самой картинки
        l = im @ np.array([0.299, 0.587, 0.114], np.float32)
        return np.clip((ghi - l) / (ghi - glo + 1e-6), 0, 1) ** 1.3

    d_base = np.asarray(Image.fromarray((density(img) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0))).astype(np.float32) / 255.0
    d_str = density(work)
    # потёки только добавляются к силуэту, стереть его они не могут
    d = np.maximum(d_base * 0.9, d_str)
    gy, gx = np.gradient(d_base)
    d = np.maximum(d, np.clip(np.hypot(gx, gy) * 6.0, 0, 1) * 0.45)

    # точечная фактура в пятнах
    blob = (smooth_noise(rng, H, W, 7) * 0.7 + smooth_noise(rng, H, W, 40) * 0.3) > (0.78 - 0.12 * amount)  # мелкий шум рвёт края пятен
    yy2, xx2 = np.mgrid[0:H, 0:W]
    thr = 0.55 * rng.random((H, W)).astype(np.float32) + 0.45 * (((yy2 + xx2 * 0.7) % 6) / 6.0)
    d = np.where(blob, np.maximum((d * 1.2 > thr).astype(np.float32), d_base * 0.6), d)
    d = d ** 0.8

    # цвет: сверху одно, снизу другое + шум
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    xx = np.linspace(0, 1, W, dtype=np.float32)[None, :, None]
    n = smooth_noise(rng, H, W, 5)[..., None]
    top, mid, bot = (np.array(pal[k], np.float32) / 255.0 for k in ("top", "mid", "bot"))
    t = np.clip(yy * 1.15 - 0.1 + (n - 0.5) * 0.6 + (xx - 0.5) * 0.3, 0, 1)
    T = np.clip(np.where(t < 0.5, top + (mid - top) * (t / 0.5), mid + (bot - mid) * ((t - 0.5) / 0.5)), 0, 1)

    # цветная кайма (каналы сдвинуты)
    s = max(2, int(3 * f))
    dd = np.stack([np.roll(d, (s, s), (0, 1)), d, np.roll(d, (-s, -s), (0, 1))], -1)
    Tdark = T * (1 - 0.5 * (d[..., None] ** 2))
    out = 1 - dd * (1 - Tdark)

    out = out * np.array([0.985, 0.975, 1.0], np.float32)  # фон чуть лавандовый
    out += rng.normal(0, 0.012, out.shape).astype(np.float32)
    out = np.clip(out, 0, 1)[s:-s, s:-s]  # срезаем края шоб кайма не лезла
    return Image.fromarray((out * 255).astype(np.uint8))


# --- запуск ---

def main():
    ap = argparse.ArgumentParser(description="Глитчкор-эффект с заполнением всего кадра шумом.")
    ap.add_argument("input", nargs="?", help="путь к картинке (если не указан — выбор при запуске)")
    ap.add_argument("-o", "--out", help="папка (или .png-файл при -n 1) для результата")
    ap.add_argument("-n", "--variants", type=int, default=1, help="сколько вариантов сделать (по умолчанию 1)")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="red", help="цветовая схема")
    ap.add_argument("--style", choices=["noise", "stretch"], default="noise",
                    help="noise - весь кадр в шум (по умолчанию), stretch - растяжка пикселей, потёки, штриховка")
    ap.add_argument("--amount", type=float, default=0.7,
                    help="сила эффекта для --style stretch (0.3 слабо, 0.7 норм, 1.0 жёстко, фигуру уже не видно)")
    ap.add_argument("--ink", type=parse_hex, help="свой цвет шумовых линий, #rrggbb")
    ap.add_argument("--accent1", type=parse_hex, help="свой акцент 1, #rrggbb")
    ap.add_argument("--accent2", type=parse_hex, help="свой акцент 2, #rrggbb")
    ap.add_argument("--seed", type=int, help="seed для повторяемого результата")
    ap.add_argument("--size", type=int, default=1080, help="длинная сторона результата в пикселях")
    ap.add_argument("--density", type=float, default=1.0, help="плотность шума (1.0 = как раньше)")
    ap.add_argument("--voids", type=float, default=0.44, help="чёрные провалы: больше = больше чёрного")
    ap.add_argument("--invert", action="store_true", help="если на исходнике светлые линии на тёмном фоне")
    ap.add_argument("--tones", action="store_true",
                    help="для цветных/залитых рисунков (не чистая линия): фон -> шум, персонаж сохраняет тона")
    ap.add_argument("--outline", action="store_true",
                    help="в режиме --tones вернуть цветную обводку вокруг персонажа (по умолчанию выкл.)")
    ap.add_argument("--bg", choices=["auto", "color", "light", "grabcut"], default="auto",
                    help="как искать фон в --tones: auto/color (по цвету краёв, тёмный и светлый), "
                         "light (только светлый, как раньше), grabcut (сложные случаи, нужен opencv)")
    ap.add_argument("--bg-tol", type=float, default=20.0,
                    help="насколько цвет может отличаться от краёв чтоб считаться фоном (больше = больше фона)")
    ap.add_argument("--rect", type=parse_rect,
                    help="где перс: x0,y0,x1,y1 от 0 до 1. всё вне прямоугольника станет шумом. пример 0.2,0,1,1")
    ap.add_argument("--mask", help="своя маска (png/jpg): белое = перс, чёрное = шум. перебивает авто-поиск")
    ap.add_argument("--show-mask", action="store_true", help="сохранить маску фона рядом (глянуть что он нашёл)")
    ap.add_argument("--no-gui", action="store_true", help="не открывать окна выбора файла, спрашивать в консоли")
    args = ap.parse_args()

    use_gui = not args.no_gui

    src = clean_path(args.input) if args.input else ask_file(use_gui)
    if not os.path.isfile(src):
        sys.exit(f"Файл не найден: {src}")

    default_dir = os.path.dirname(os.path.abspath(src))
    out_file = None
    if args.out:
        o = clean_path(args.out)
        if o.lower().endswith(EXTS):
            out_file = o
            out_dir = os.path.dirname(os.path.abspath(o))
        else:
            out_dir = o
    elif args.input:
        out_dir = default_dir  # путь дали - сохраняем рядом
    else:
        out_dir = ask_folder(default_dir, use_gui)
    os.makedirs(out_dir, exist_ok=True)

    colors = dict(PRESETS[args.preset])
    if args.ink:
        colors["ink"] = args.ink
    if args.accent1:
        colors["a1"] = args.accent1
    if args.accent2:
        colors["a2"] = args.accent2

    ink = tones = bg = None
    if args.style == "noise":
        ink = load_ink(src, args.size, args.invert)
    if args.style == "noise" and args.tones:
        tones, bg = load_tones(src, args.size, args.bg, args.bg_tol, args.rect,
                               clean_path(args.mask) if args.mask else None)
    stem = os.path.splitext(os.path.basename(src))[0]
    base_seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(2), "big")
    rgb = None
    if args.style == "stretch":
        rgb = np.asarray(_load_img(src, args.size)).astype(np.float32) / 255.0
    if args.show_mask and bg is not None:
        mp = os.path.join(out_dir, f"{stem}_mask.png")
        Image.fromarray((~bg).astype(np.uint8) * 255).save(mp)  # белое = перс
        print(f"Маска: {mp}")

    for i in range(max(1, args.variants)):
        seed = base_seed + i
        if args.style == "stretch":
            img = render_stretch(rgb, seed, STRETCH_PALS[args.preset], args.amount)
        else:
            img = render(ink, seed, colors, args.density, args.voids, tones, bg, args.outline)
        if out_file and args.variants == 1:
            path = out_file
        else:
            tag = "stretch" if args.style == "stretch" else "glitch"
            path = os.path.join(out_dir, f"{stem}_{tag}_{args.preset}_{seed}.png")
        img.save(path)
        print(f"Готово: {path}  (seed {seed})")


if __name__ == "__main__":
    main()
