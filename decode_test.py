import os
import time

from src.detect_image import load_detection_model, detect_and_process
# from src.decode import TrOCRDecoder
from src.decode_ctc import CTCDecoder

# Import stałych
from src.config import (
    DECODE_TEST_INPUT_DIR, DECODE_TEST_MAX_IMAGES, DECODE_TEST_OUTPUT_RESULTS, DECODE_TEST_OUTPUT_TIMES,
    DATE_MODEL_PATH, CODE_MODEL_PATH, DEBUG_MODE
)

# === KONFIGURACJA ===

INPUT_DIR = DECODE_TEST_INPUT_DIR
MAX_IMAGES = DECODE_TEST_MAX_IMAGES

OUTPUT_RESULTS = DECODE_TEST_OUTPUT_RESULTS
OUTPUT_TIMES = DECODE_TEST_OUTPUT_TIMES

date_model_path = DATE_MODEL_PATH
code_model_path = CODE_MODEL_PATH

# === ŁADOWANIE MODELI ===

print("Ładowanie modeli...")
start_global = time.perf_counter()

# YOLOv5 model
model, stride, names, pt = load_detection_model()

# OCR model
# ocr = TrOCRDecoder()
ocr_date = CTCDecoder(date_model_path, text_type="date", device="cpu")
ocr_code = CTCDecoder(code_model_path, text_type="code", device="cpu")

total_model_load_time = time.perf_counter() - start_global
print(f"Modele załadowane w {total_model_load_time:.2f} s")

# === PRZETWARZANIE ZDJĘĆ ===

files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
files = sorted(files)
if MAX_IMAGES:
    files = files[0:MAX_IMAGES]

print(f"Przetwarzam {len(files)} zdjęć...")

with open(OUTPUT_RESULTS, "w") as res_file, open(OUTPUT_TIMES, "w") as time_file:
    for idx, filename in enumerate(files):
        filepath = os.path.join(INPUT_DIR, filename)
        print(f"[{idx+1}/{len(files)}] {filename}")

        try:            # DETEKCJA
            t_start = time.perf_counter()
            result_img, detect_time, crops_top, crops_bottom, crop_time = detect_and_process(filepath, model, stride,
                                                                                             names, pt, DEBUG_MODE)

            total_detect_crop_time = time.perf_counter() - t_start

            # OCR
            decoded_msg_date, preproc_time_date, decode_time_date, decode_total_time_date = ocr_date.decode_ufi(
                crops_top[0], text_type="date", return_times=True)
            decoded_msg_code, preproc_time_code, decode_time_code, decode_total_time_code = ocr_code.decode_ufi(
                crops_bottom[0],text_type="code", return_times=True)

            total_time = time.perf_counter() - t_start
            # zapisz wyniki
            res_file.write(f"{filename} {decoded_msg_date} {decoded_msg_code}\n")
            time_file.write(
                f"{filename} Detect: {total_detect_crop_time:.3f} "
                f"Decode_date: {decode_time_date:.3f} "
                f"Decode code: {decode_time_code:.3f} "
                f"Detect + decode: {total_detect_crop_time+decode_time_date+decode_time_code} "
                f"Total {total_time}\n"
            )
            res_file.flush()
            time_file.flush()

        except Exception as e:
            print(f"Błąd przetwarzania pliku {filename}: {e}")
            res_file.write(f"{filename}\tERROR\n")
            time_file.write(f"{filename}\tERROR\n")
            res_file.flush()
            time_file.flush()

print("Przetwarzanie zakończone.")
