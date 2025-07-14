import sys
import cv2
import os
import time
from pathlib import Path
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

# import funkcji
from src.image_preprocessing import processing_image, calculate_threshold_value_based_on_green
from src.detect_image import load_detection_model, predict
from src.my_utils.PP_search import find_angle

# import stałych
from src.config import (
    ROI_TEST_TYPE,
    ROI_INPUT_DIR, ROI_OUTPUT_DIR, ROI_THRESHOLD_TEST_DIR,
    ROI_LOWER_RANGE, ROI_UPPER_RANGE, OFFSET_DOWN, DETECT_DEVICE, DO_PP_ROTATE, DEBUG_MODE
)


# import YOLOv5 jako moduł
sys.path.append(str(Path(__file__).resolve().parents[1] / "yolov5"))
from models.common import DetectMultiBackend  # Import backendu modelu


def generate_roi_data(model_path="best.pt", debug=DEBUG_MODE):
    #           === SETTINGS ===
    test_type = ROI_TEST_TYPE           # 1 - generate crops
                                        # 2 - test threshold offset
                                        # 3 - test hue threshold values
    IMAGE_PATH = ROI_INPUT_DIR
    OUTPUT_PATH = ROI_OUTPUT_DIR

    lower_range = ROI_LOWER_RANGE         # default = 2
    upper_range = ROI_UPPER_RANGE         # default = 138
    NUM_IMAGES_IN_FOLDER = 1000            # default = 0
    offset = OFFSET_DOWN                  # default = -13

    image_name = "88"                            # test image name
    image_path = "/all_images/image0000088.jpg"  # test image path

    model, stride, names, pt = load_detection_model()
    device = DETECT_DEVICE

    # if test_type == 1:      # setting default test 1 values
    #     lower_range = 2
    #     upper_range = 138
    #     offset = -13
        # rejected images: 20, 109, 110, 123, 130

    for i in range(lower_range, upper_range+1):
        if test_type == 1:
            # image_name = f"image{str(i).zfill(7)}.jpg"
            if i < 10:
                image_name = f"img_000{str(i)}.jpg"
            elif i < 100:
                image_name = f"img_00{str(i)}.jpg"
            elif i < 1000:
                image_name = f"img_0{str(i)}.jpg"
            else:
                image_name = f"img_{str(i)}.jpg"
            image_path = os.path.join(IMAGE_PATH, image_name).replace("\\", "/")
        print(f"image_path: {image_path}")
        # Sprawdź, czy plik obrazu istnieje
        if not Path(image_path).exists():
            print(f"Plik {image_path} nie istnieje!")
            return

        # Wczytaj obraz
        start = time.perf_counter()
        image = cv2.imread(image_path)
        if image is None:
            print(f"Nie można wczytać obrazu: {image_path}")
            return

        if test_type == 2:
            offset += 1
        threshold_value = calculate_threshold_value_based_on_green(image_path, offset)

        # Dokonaj predykcji
        pred, img, pred_time = predict(image, model, stride, 0)

        # Znajdź kąt "produkt polski"
        pp_rotate = DO_PP_ROTATE
        label_angle = 0
        if pp_rotate:
            label_angle, pp_time = find_angle(str(image_path))
            # print(f"label_angle: {label_angle}")

        # Przeprocesuj obraz
        result_crops_top = []
        result_crops_bottom = []
        for det in pred:
            if len(det):
                result_image_top, result_image_bottom = processing_image(image, img, det, threshold_value, label_angle, pp_rotate, i)

                result_crops_top.append(result_image_top)
                result_crops_bottom.append(result_image_bottom)

        result_crop_top = result_crops_top[0]
        result_crop_bottom = result_crops_bottom[0]

        if test_type == 1:
            os.makedirs(OUTPUT_PATH, exist_ok=True)
            generated_data_path_top = os.path.join(OUTPUT_PATH, f"date/").replace("/", "\\")
            os.makedirs(generated_data_path_top, exist_ok=True)
            if i < 10:
                generated_data_path_top = os.path.join(generated_data_path_top, f"img_000{i}.jpg")
            elif i < 100:
                generated_data_path_top = os.path.join(generated_data_path_top, f"img_00{i}.jpg")
            elif i < 1000:
                generated_data_path_top = os.path.join(generated_data_path_top, f"img_0{i}.jpg")
            else:
                generated_data_path_top = os.path.join(generated_data_path_top, f"img_{i}.jpg")
            cv2.imwrite(generated_data_path_top, result_crop_top)
            print(f"Zapisano: {generated_data_path_top}")

            generated_data_path_bottom = os.path.join(OUTPUT_PATH, f"code/").replace("/", "\\")
            os.makedirs(generated_data_path_bottom, exist_ok=True)
            if i < 10:
                generated_data_path_bottom = os.path.join(generated_data_path_bottom, f"img_000{i}.jpg")
            elif i < 100:
                generated_data_path_bottom = os.path.join(generated_data_path_bottom, f"img_00{i}.jpg")
            elif i < 1000:
                generated_data_path_bottom = os.path.join(generated_data_path_bottom, f"img_0{i}.jpg")
            else:
                generated_data_path_bottom = os.path.join(generated_data_path_bottom, f"img_{i}.jpg")
            cv2.imwrite(generated_data_path_bottom, result_crop_bottom)
            print(f"Zapisano: {generated_data_path_bottom}")
        if test_type == 2:
            os.makedirs(ROI_THRESHOLD_TEST_DIR, exist_ok=True)
            generated_data_path_top = os.path.join(ROI_THRESHOLD_TEST_DIR, f"date/").replace("/", "\\")
            generated_data_path_top = os.path.join(generated_data_path_top, f"img_{image_name}_with_{offset}_offset.jpg")
            cv2.imwrite(generated_data_path_top, result_crop_top)
            print(f"Zapisano: {generated_data_path_top}")

            generated_data_path_bottom = os.path.join(ROI_THRESHOLD_TEST_DIR, f"code/").replace("/", "\\")
            generated_data_path_bottom = os.path.join(generated_data_path_bottom, f"img_{image_name}_with_{offset}_offset.jpg")
            cv2.imwrite(generated_data_path_bottom, result_crop_bottom)
            print(f"Zapisano: {generated_data_path_bottom}")

    return 0


if __name__ == "__main__":
    generate_roi_data()
    input("Naciśnij Enter, aby zakończyć...")

