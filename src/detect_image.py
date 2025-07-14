import sys
import time
import cv2
import numpy as np
import torch
import screeninfo
from pathlib import Path
import pathlib

# Obsługa WindowsPath
pathlib.PosixPath = pathlib.WindowsPath

# Import funkcji
from src.image_preprocessing import processing_image, calculate_threshold_value_based_on_green
from src.my_utils.my_utils import plot_one_box
from src.my_utils.PP_search import find_angle

# Import konfiguracji projektu
from src.config import (
    MODEL_PATH, CONFIDENCE_THRESHOLD, IOU_THRESHOLD,
    DEBUG_MODE, DISPLAY_SCALE_FACTOR, DO_PP_ROTATE, DETECT_DEVICE
)

# Importy YOLOv5
sys.path.append(str(Path(__file__).resolve().parents[1] / "yolov5"))
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from models.common import non_max_suppression


def detect_and_process(image_path, model, stride, names, pt, debug=DEBUG_MODE):
    # Sprawdzenie istnienia obrazu
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Plik {image_path} nie istnieje!")
        return

    # Wczytanie obrazu
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Nie można wczytać obrazu: {image_path}")
        return

    # Obliczenie progu progowania
    threshold_value = calculate_threshold_value_based_on_green(str(image_path))

    # Wyznaczenie kąta etykiety (PP_search)
    label_angle, pp_search_time = find_angle(str(image_path), DEBUG_MODE)
    if debug:
        print(f"Label angle: {label_angle}")

    # # Wczytanie modelu YOLOv5
    # model, stride, names, pt = load_detection_model()

    # Predykcja detekcji
    start_time = time.perf_counter()
    pred, img, pred_time = predict(image, model, stride, start_time)

    result_crops_top = []
    result_crops_bottom = []

    crop_time_start = time.perf_counter()
    # Przetwarzanie wyników detekcji
    for det in pred:
        if len(det):
            result_image_top, result_image_bottom = processing_image(image, img, det, threshold_value, label_angle,
                                                                     pp_rotate=DO_PP_ROTATE, i=1, debug=DEBUG_MODE)
            result_crops_top.append(result_image_top)
            result_crops_bottom.append(result_image_bottom)

            if debug:
                for d in det:
                    x1, y1, x2, y2, conf, cls = d[:6]
                    label = f"{names[int(cls.item())]} {conf.item():.2f}"
                    plot_one_box([x1.item(), y1.item(), x2.item(), y2.item()], image, label=label, color=(0, 255, 0), line_thickness=2)
    crop_time_stop = time.perf_counter()
    crop_time = crop_time_stop - crop_time_start

    # Wyświetlanie wynikowego obrazu (skalowanie pod ekran)
    show_result(image, pred_time)

    return image, pred_time, result_crops_top, result_crops_bottom, crop_time


def load_detection_model(model_path=MODEL_PATH, device=DETECT_DEVICE):
    model = DetectMultiBackend(model_path, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    return model, stride, names, pt


def predict(image, model, stride, start_time):
    # Przygotowanie obrazu do detekcji (letterbox → tensor → normalizacja)
    img = letterbox(image, stride=stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC -> CHW -> RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float() / 255.0
    img = img.unsqueeze(0)

    pred_raw = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred_raw[0], conf_thres=CONFIDENCE_THRESHOLD, iou_thres=IOU_THRESHOLD)
    pred_time = time.perf_counter() - start_time
    return pred, img, pred_time


# def predict(image, model, stride, start_time):    # pod detect test
#     original_shape = image.shape[:2]  # (h, w)
#
#     # letterbox (bez return_info)
#     img = letterbox(image, stride=stride, auto=True)[0]  # [0] bo zwraca tylko obraz
#     resized_shape = img.shape[:2]  # (h, w)
#
#     # Oblicz ratio i padding (zgodnie z YOLOv5 letterbox)
#     ratio = (resized_shape[1] / original_shape[1], resized_shape[0] / original_shape[0])  # (w, h)
#     pad_w = (resized_shape[1] - original_shape[1] * ratio[0]) / 2
#     pad_h = (resized_shape[0] - original_shape[0] * ratio[1]) / 2
#     pad = (pad_w, pad_h)
#
#     # Przygotowanie do modelu
#     img = img.transpose((2, 0, 1))[::-1]  # HWC -> CHW -> RGB
#     img = np.ascontiguousarray(img)
#     img = torch.from_numpy(img).float() / 255.0
#     img = img.unsqueeze(0)
#
#     pred_raw = model(img, augment=False, visualize=False)
#     pred = non_max_suppression(pred_raw[0], conf_thres=CONFIDENCE_THRESHOLD, iou_thres=IOU_THRESHOLD)
#     pred_time = time.perf_counter() - start_time
#
#     return pred, img, pred_time, ratio, pad


def show_result(image, pred_time, debug=DEBUG_MODE):
    screen = screeninfo.get_monitors()[0]
    screen_width, screen_height = screen.width, screen.height
    img_height, img_width = image.shape[:2]
    scale_factor = min(screen_width / img_width, screen_height / img_height, 1.0) * DISPLAY_SCALE_FACTOR
    resized_image = cv2.resize(image, (int(img_width * scale_factor), int(img_height * scale_factor)))

    if debug:
        print(f"Czas detekcji: {pred_time:.6f} sekund")
        cv2.imshow("Wykryte obiekty", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "C:/Users/krukw/PycharmProjects/Baumer_test/test_images/test_image3.jpg"
    model, stride, names, pt = load_detection_model()
    detect_and_process(image_path, model, stride, names, pt)
    input("Naciśnij Enter, aby zakończyć...")
