from __future__ import annotations

from pathlib import Path
from PIL import Image

import cv2
import gzip
import numpy as np


def resize(x: np.ndarray, size: int) -> np.ndarray:
    h, w, c = x.shape
    s = size / min(h, w)
    return cv2.resize(x, (int(w * s), int(h * s)))


def circshift(x: np.ndarray, shift: np.ndarray) -> np.ndarray:
    for i in range(shift.size):
        x = np.roll(x, shift[i], axis=i)
    return x


def prepare_psf(psf: np.ndarray, out_size: tuple) -> np.ndarray:
    psf_size = np.int32(psf.shape)
    out_size = psf_size if out_size is None else np.int32(out_size)
    new_psf = np.zeros(out_size, dtype=np.float32)
    new_psf[:psf_size[0], :psf_size[1]] = psf[:, :]
    return circshift(new_psf, -psf_size // 2)


def psf2otf(psf: np.ndarray, out_size: tuple) -> np.ndarray:
    return np.complex64(np.fft.fftn(prepare_psf(psf, out_size)))


def l0_smoothing(x: np.ndarray, kappa: float = 2.0, lamb: float = 1e-1) -> np.ndarray:
    S = x / 255.0
    H, W, C = x.shape
    
    size_2d = [H, W]
    fx = np.int32([[1, -1]])
    fy = np.int32([[1], [-1]])
    otffx = psf2otf(fx, size_2d)
    otffy = psf2otf(fy, size_2d)

    FI = np.complex64(np.zeros((H, W, C)))
    FI[..., 0] = np.fft.fft2(S[..., 0])
    FI[..., 1] = np.fft.fft2(S[..., 1])
    FI[..., 2] = np.fft.fft2(S[..., 2])

    MTF = np.abs(otffx) ** 2 + np.abs(otffy) ** 2
    MTF = np.tile(MTF[..., None], (1, 1, C))

    h    = np.zeros((H, W, C), dtype=np.float32)
    v    = np.zeros((H, W, C), dtype=np.float32)
    dxhp = np.zeros((H, W, C), dtype=np.float32)
    dyvp = np.zeros((H, W, C), dtype=np.float32)
    FS   = np.zeros((H, W, C), dtype=np.complex64)

    beta_max, beta = 1e5, 2 * lamb
    while beta < beta_max:
        h[:, 0:W-1, :] = np.diff(S, 1, 1)
        np.subtract(S[:, 0:1, :], S[:, W-1:W, :], out=h[:, W-1:W, :])

        v[0:H-1, :, :] = np.diff(S, 1, 0)
        np.subtract(S[0:1, :, :], S[H-1:H, :, :], out=v[H-1:H, :, :])

        t = np.sum(h ** 2 + v ** 2, axis=-1) < lamb / beta
        t = np.tile(t[:, :, None], (1, 1, 3))

        h[t], v[t] = 0, 0

        np.subtract(h[:, W-1:W, :], h[:, 0:1, :], out=dxhp[:, 0:1, :])
        dxhp[:, 1:W, :] = -(np.diff(h, 1, 1))
        np.subtract(v[H-1:H, :, :], v[0:1, :, :], out=dyvp[0:1, :, :])
        dyvp[1:H, :, :] = -(np.diff(v, 1, 0))
        normin = dxhp + dyvp

        FS[..., 0] = np.fft.fft2(normin[..., 0])
        FS[..., 1] = np.fft.fft2(normin[..., 1])
        FS[..., 2] = np.fft.fft2(normin[..., 2])

        denorm = 1 + beta * MTF
        FS[...] = (FI + beta * FS) / denorm

        S[..., 0] = np.float32((np.fft.ifft2(FS[..., 0])).real)
        S[..., 1] = np.float32((np.fft.ifft2(FS[..., 1])).real)
        S[..., 2] = np.float32((np.fft.ifft2(FS[..., 2])).real)

        beta *= kappa

    return (np.clip(S, 0, 1) * 255.0).astype(np.uint8)


def dog(x: np.ndarray, size: tuple[int, int], sigma: float, k: float, gamma: float) -> np.ndarray:
    if x.shape[-1] == 3: x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    return cv2.GaussianBlur(x, size, sigma) - gamma * cv2.GaussianBlur(x, size, sigma * k)


def xdog(x: np.ndarray, sigma: float, k: float, gamma: float, epsilon: float, phi: float) -> np.ndarray:
    x = dog(x, (0, 0), sigma, k, gamma) / 255
    return np.where(x < epsilon, 255, 255 * (1 + np.tanh(phi * x))).astype(np.uint8)


def sketch(x: np.ndarray, sigma: float, k: float, gamma: float, epsilon: float, phi: float, area_min: int) -> np.ndarray:
    x = xdog(x, sigma, k, gamma, epsilon, phi)
    x = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(x, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    contours = [contour for contour in contours if cv2.contourArea(contour) > area_min]
    return 255 - cv2.drawContours(np.zeros_like(x), contours, -1, (255, 255, 255), -1)


def prepare(path: Path, size: int) -> np.ndarray:
    x = resize(np.array(Image.open(path).convert("RGB")), size)
    buffer = np.zeros((*x.shape[:2], (1 * 3) + (3 * 1) + (1 * 3)), np.uint8)
    buffer[..., :3] = x
    buffer[..., 3] = sketch(x, 0.3, 4.5, 0.95, -1.0, 10e9, 2)
    buffer[..., 4] = sketch(x, 0.4, 4.5, 0.95, -1.0, 10e9, 2)
    buffer[..., 5] = sketch(x, 0.5, 4.5, 0.95, -1.0, 10e9, 2)
    buffer[..., 6:] = l0_smoothing(x, 2, 0.1)
    return buffer


def process(path: Path, outdir: Path, size: int) -> None:
    with gzip.GzipFile(outdir / Path(f"{path.name}.npy.gz"), "wb") as f:
        np.save(f, prepare(path, size))


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    from tqdm import tqdm
    

    outdir = Path("prepared")
    outdir.mkdir(exist_ok=True)
    
    for split in ["train", "test", "val"]:
        splitdir = outdir / Path(split)
        splitdir.mkdir(exist_ok=True)

        paths = list((Path("raw") / Path(split)).glob("*.png"))
        fn = partial(process, outdir=splitdir, size=512)
        with ProcessPoolExecutor() as executor:
            results = executor.map(fn, paths)
            list(tqdm(results, desc="Preparing Data", total=len(paths)))