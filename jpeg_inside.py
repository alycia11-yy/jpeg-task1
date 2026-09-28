"""Task 1: Look inside a JPEG.
Usage: python jpeg_inside.py photo.jpg
Needs: pip install pillow numpy scipy matplotlib
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import dctn

path = sys.argv[1] if len(sys.argv) > 1 else "photo.jpg"


def luma(im):
    """Y channel as float array, cropped to a multiple of 8."""
    y = np.asarray(im.convert("YCbCr"), dtype=np.float64)[..., 0]
    h, w = (y.shape[0] // 8) * 8, (y.shape[1] // 8) * 8
    return y[:h, :w]


def block_dct(y):
    """8x8 block DCT of a level-shifted image. Returns (H/8, W/8, 8, 8)."""
    h, w = y.shape
    blocks = (y - 128).reshape(h // 8, 8, w // 8, 8).swapaxes(1, 2)
    return dctn(blocks, axes=(2, 3), norm="ortho")


def coeff_map(d):
    """Tile the blocks back into an image layout for display."""
    bh, bw = d.shape[:2]
    return d.swapaxes(1, 2).reshape(bh * 8, bw * 8)


# --- 1. Quantization table ---------------------------------------------------
im = Image.open(path)
for k, table in im.quantization.items():
    print(f"Table {k} (0 = luma, 1 = chroma):")
    print(np.array(table).reshape(8, 8), "\n")

# --- 2. DCT of each 8x8 block -------------------------------------------------
y = luma(im)
d = block_dct(y)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(np.log1p(np.abs(coeff_map(d))), cmap="magma")
ax[0].set_title("log|DCT| of every block")
m = ax[1].imshow(np.log1p(np.abs(d).mean(axis=(0, 1))), cmap="magma")
ax[1].set_title("mean log|coef| per frequency\n(top-left = DC)")
plt.colorbar(m, ax=ax[1])
plt.tight_layout()
plt.savefig("dct_coeffs.png", dpi=150)

# --- 3. Re-save at another quality and compare -------------------------------
src = Image.open(path).convert("RGB")
src.save("q95.jpg", quality=95)
Image.open("q95.jpg").save("q60.jpg", quality=60)

a, b = Image.open("q95.jpg"), Image.open("q60.jpg")
ya, yb = luma(a), luma(b)
pix_diff = np.abs(ya - yb)
dct_diff = np.abs(block_dct(ya) - block_dct(yb))

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
ax[0].imshow(pix_diff, cmap="gray", vmax=np.percentile(pix_diff, 99))
ax[0].set_title("|pixel diff| (Y), 95 vs 60")
ax[1].imshow(np.log1p(coeff_map(dct_diff)), cmap="magma")
ax[1].set_title("log|DCT diff| per block")
m = ax[2].imshow(dct_diff.mean(axis=(0, 1)), cmap="magma")
ax[2].set_title("mean |DCT diff| per frequency")
plt.colorbar(m, ax=ax[2])
plt.tight_layout()
plt.savefig("dct_diff.png", dpi=150)

print("q95 table:\n", np.array(a.quantization[0]).reshape(8, 8))
print("q60 table:\n", np.array(b.quantization[0]).reshape(8, 8))
plt.show()
