import os
import cv2
import time
from PIL import Image
from PIL import ImageDraw

from src.detect_image import load_detection_model, predict
from src.my_utils.my_utils import plot_one_box
from src.config import (
    DETECT_TEST_INPUT_DIR, DETECT_TEST_MAX_IMAGES, DETECT_TEST_OUTPUT_RESULTS, DEBUG_MODE
)

# === KONFIGURACJA ===

INPUT_DIR = DETECT_TEST_INPUT_DIR
MAX_IMAGES = DETECT_TEST_MAX_IMAGES
OUTPUT_RESULTS = DETECT_TEST_OUTPUT_RESULTS

# === ŁADOWANIE MODELU ===

print("Ładowanie modelu...")
start_global = time.perf_counter()

model, stride, names, pt = load_detection_model()

total_model_load_time = time.perf_counter() - start_global
print(f"Modele załadowane w {total_model_load_time:.2f} s")

# === FUNKCJE ===

def get_center(pred, ratio, pad):
    if not pred or len(pred[0]) < 1:
        raise ValueError("Brak danych w predykcji.")

    x1, y1, x2, y2, conf, cls = pred[0][0][:6]

    # Odwrócenie paddingu i skalowania
    x1 = (x1 - pad[0]) / ratio[0]
    y1 = (y1 - pad[1]) / ratio[1]
    x2 = (x2 - pad[0]) / ratio[0]
    y2 = (y2 - pad[1]) / ratio[1]

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    center = (int(center_x.item()), int(center_y.item()))
    return center, (x1.item(), y1.item(), x2.item(), y2.item())


def draw_debug(img, pt, label):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.ellipse((pt[0]-4, pt[1]-4, pt[0]+4, pt[1]+4), fill="blue")
    draw.text((pt[0]+6, pt[1]-12), label, fill="blue")
    return img_pil

# === PRZETWARZANIE ZDJĘĆ ===

files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
files = sorted(files)
if MAX_IMAGES:
    files = files[:MAX_IMAGES]

print(f"Przetwarzam {len(files)} zdjęć...")

with open(OUTPUT_RESULTS, "w") as res_file:
    for idx, filename in enumerate(files):
        filepath = os.path.join(INPUT_DIR, filename)
        print(f"[{idx+1}/{len(files)}] {filename}")

        try:
            t_start = time.perf_counter()

            image = cv2.imread(str(filepath))
            if image is None:
                print(f"Nie można wczytać obrazu: {filepath}")
                break

            t_pred_start = time.perf_counter()
            pred, img, pred_time, ratio, pad = predict(image, model, stride, t_pred_start)
            total_time = time.perf_counter() - t_start

            # print(f"pred: {pred}\npred[0]: {pred[0]}")
            center, (x1, y1, x2, y2) = get_center(pred, ratio, pad)

            if DEBUG_MODE:
                print(f"text_cords: {center}")
                plot_one_box([x1, y1, x2, y2], image, label="pred", color=(0, 255, 0), line_thickness=2)
                debug_img = draw_debug(image, center, "pred")
                debug_img.show()

            res_file.write(f"{filename} {center} {pred_time}\n")
            res_file.flush()

        except Exception as e:
            print(f"Błąd przetwarzania pliku {filename}: {e}")
            res_file.write(f"{filename}\tERROR\n")
            res_file.flush()

print("Przetwarzanie zakończone.")