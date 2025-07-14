from pathlib import Path
import re
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

# ─────────────────────────────── Configuration ──────────────────────────────
INPUT_DIR = Path(
    r"C:\Users\krukw\PycharmProjects\Baumer_test\pom\in"
)
OUTPUT_DIR = Path(
    r"C:\Users\krukw\PycharmProjects\Baumer_test\pom\out"
)

WIN_WIDTH = 5  # window width  (can be even or odd >=1)
WIN_HEIGHT = 15  # window height (ca be even or odd >=1)
debug = False

NUM_WORKERS = 0  # 0 → serial, >0 → parallel threads (recommended CPU-cores-1)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ─────────────────────────────── Helper utilities ───────────────────────────

def natural_sort_key(path: Path):
    """Return a key so that filenames with numbers sort naturally."""
    return [int(ch) if ch.isdigit() else ch.lower() for ch in re.split(r"([0-9]+)", path.name)]


# ───────────────────────────── Sliding-window mean ──────────────────────────

def sliding_window_mean(img: np.ndarray, win_w: int, win_h: int) -> np.ndarray:
    if img.ndim != 2:
        raise ValueError("sliding_window_mean expects a 2-D greyscale image")
    if win_w == 1 and win_h == 1:
        return img.copy()

    half_w, half_h = win_w // 2, win_h // 2

    # Compute box-filtered mean for the whole image in float32
    mean_img = cv2.blur(img.astype(np.float32), (win_w, win_h))

    # Build a boolean mask for valid replacement points
    mask = img != 0  # skip original black pixels
    if half_h:
        mask[:half_h, :] = False
        mask[-half_h:, :] = False
    if half_w:
        mask[:, :half_w] = False
        mask[:, -half_w:] = False

    out = img.copy()
    out[mask] = mean_img[mask].astype(np.uint8)
    return out


# ───────────────────────────── Image processing ─────────────────────────────

def process_single_image_folder(path: Path) -> str:  # do robienia całego folderu
    """Load, process and save a single image; return filename or error msg."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return f"SKIP {path.name} (cannot read)"

    processed = sliding_window_mean(img, WIN_WIDTH, WIN_HEIGHT)
    cv2.imwrite(str(OUTPUT_DIR / path.name), processed)

    if debug:
        collage = cv2.hconcat([img, processed])
        collage = cv2.resize(collage, (1000, 700), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Original | Processed", collage)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

    return f"OK  {path.name}"

def process_single_image(img):
    processed = sliding_window_mean(img, WIN_WIDTH, WIN_HEIGHT)

    return processed


# ─────────────────────────────────── Main ───────────────────────────────────

def gather_image_paths() -> List[Path]:
    return sorted(
        [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS],
        key=natural_sort_key,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img_paths = gather_image_paths()

    if not img_paths:
        print(f"No images found in {INPUT_DIR.resolve()}")
        return

    total = len(img_paths)
    print(f"Found {total} image(s). Processing…\n")

    if NUM_WORKERS > 1:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
            for i, res in enumerate(pool.map(process_single_image_folder, img_paths), 1):
                print(f"[{i}/{total}] {res}")
    else:
        for i, p in enumerate(img_paths, 1):
            res = process_single_image_folder(p)
            print(f"[{i}/{total}] {res}")

    print("\n✔ All images processed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Program przerwany przez użytkownika. Zamykanie…")
