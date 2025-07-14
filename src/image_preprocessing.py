import sys
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path

# Import funkcji narzędziowych
from src.my_utils.my_utils import debug_crop
from src.my_utils.blurre_image import process_single_image

# Import konfiguracji
from src.config import (
    BLUE_HUE_LOWER, BLUE_HUE_UPPER, BLUE_SAT_LOWER, BLUE_SAT_UPPER, BLUE_LIG_LOWER, BLUE_LIG_UPPER,
    MAX_BLOB_AREA, PCA_ANGLE_OFFSET, PADDING_HEIGHT_RATIO, CENTER_TARGET_SIZE,
    DIVIDE_IMAGE_THRESHOLD, RECT_SIZE, RECT_OFFSET, OFFSET_DOWN,
    MIN_BLACK_PIXELS, TO_BLACK_MARGIN,
    GREEN_COEF_A, GREEN_COEF_B, LABEL_CENTER_PROP,
    THRESHOLD_RAGE, THRESHOLD_WINDOW_SIZE, GREEN_CHANNEL_ONLY,
    DEBUG_MODE
)

# Import funkcji YOLOv5 do skalowania bboxów
sys.path.append(str(Path(__file__).resolve().parents[1] / "yolov5"))
from utils.general import scale_boxes


# ======================= GŁÓWNA FUNKCJA PRZETWARZANIA =======================

def processing_image(image, img, det, threshold_value, label_angle, pp_rotate, i, debug=DEBUG_MODE):
    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], image.shape).round()

    # if pp_rotate:
    #     image, det = PP_rotate(image, label_angle, det)

    crop = crop_detected_regions(image, det)
    crop = resize_crop(crop)

    if pp_rotate:
        pp_rotated_crop = rotate_image(crop, label_angle)
    else:
        pp_rotated_crop = crop

    blue_mask = threshold_hue(pp_rotated_crop, BLUE_HUE_LOWER, BLUE_HUE_UPPER,
                               BLUE_SAT_LOWER, BLUE_SAT_UPPER, BLUE_LIG_LOWER, BLUE_LIG_UPPER)
    non_blue_mask = cv2.bitwise_not(blue_mask)
    white_background = np.full_like(pp_rotated_crop, 255)
    masked_blue = cv2.bitwise_and(pp_rotated_crop, pp_rotated_crop, mask=non_blue_mask)
    masked_blue = cv2.bitwise_or(masked_blue, cv2.bitwise_and(white_background, white_background, mask=blue_mask))

    green_only = masked_blue[:, :, 1]
    _, thresholded = cv2.threshold(green_only, threshold_value, 255, cv2.THRESH_BINARY)
    # thresholded = adaptive_hist_threshold(masked_blue, threshold_value)
    filtered = filter_large_blobs(thresholded, i, debug=debug)

    rotated, angle = correct_angle(filtered, debug=debug)
    padded = pad_image(rotated, PADDING_HEIGHT_RATIO)
    rotated2, angle2 = correct_angle(padded, debug=debug)
    divided_top, divided_bottom = divide_image(rotated2)

    cropped_divided_top = crop_to_black_content(divided_top)
    cropped_divided_bottom = crop_to_black_content(divided_bottom)

    centered_top = center_image(cropped_divided_top)
    centered_bottom = center_image(cropped_divided_bottom)

    blurred_top = process_single_image(centered_top)
    blurred_bottom = process_single_image(centered_bottom)

    if debug:
        debug_images = [crop, pp_rotated_crop, blue_mask, masked_blue, filtered, thresholded, rotated,
                        padded, rotated2, divided_top, divided_bottom,
                        cropped_divided_top, cropped_divided_bottom, centered_top, centered_bottom]
        debug_names = ["crop", "pp_rotated_crop", "blue_mask", "masked_blue", "filtered", "thresholded", "rotated",
                       "padded", "rotated2", "divided_top", "divided_bottom",
                       "cropped_divided_top", "cropped_divided_bottom", "centered_top", "centered_bottom"]
        debug_crop(debug_images, debug_names, angle, angle2)

    return blurred_top, blurred_bottom


# ======================= POMOCNICZE FUNKCJE =======================

def resize_crop(crop):
    new_size = (crop.shape[1] * 2, crop.shape[0] * 2)
    crop = Image.fromarray(crop).resize(new_size, Image.BICUBIC)
    return np.array(crop)


def crop_detected_regions(image, detections):
    x1, y1, x2, y2 = map(int, detections[0][:4])
    return image[y1:y2, x1:x2].copy()


def threshold_hue(img_bgr, lower_hue, upper_hue, lower_sat, upper_sat, lower_lig, upper_lig):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([lower_hue // 2, lower_sat, lower_lig])
    upper = np.array([upper_hue // 2, upper_sat, upper_lig])
    return cv2.inRange(img_hsv, lower, upper)


def compute_centroid(binary_image):
    black_pixels = np.column_stack(np.where(binary_image == 0))
    if len(black_pixels) == 0:
        return None
    centroid_y = int(np.mean(black_pixels[:, 0]))
    centroid_x = int(np.mean(black_pixels[:, 1]))
    return centroid_y, centroid_x


def filter_large_blobs(img, img_id=None, max_area=MAX_BLOB_AREA, debug=DEBUG_MODE):
    img_copy = cv2.bitwise_not(img.copy())
    contours, _ = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        debug_img = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        for i, cnt in enumerate(contours_sorted[:3]):
            area = cv2.contourArea(cnt)
            cv2.drawContours(debug_img, [cnt], -1, colors[i], 2)
            x, y, w, h = cv2.boundingRect(cnt)
            text_pos = (x, y - 100 if y - 100 > 100 else y + 100)
            cv2.putText(debug_img, f"{int(area)}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)
        cv2.imshow(f"Contours Areas (img {img_id})", debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    for cnt in contours:
        if cv2.contourArea(cnt) > max_area:
            cv2.drawContours(img_copy, [cnt], -1, 0, thickness=cv2.FILLED)

    return cv2.bitwise_not(img_copy)


def correct_angle(image, debug=DEBUG_MODE):
    coords = np.column_stack(np.where(image == 0))
    if len(coords) < 10:
        print("Za mało punktów do PCA.")
        return image, 0

    mean, eigenvectors = cv2.PCACompute(coords.astype(np.float32), mean=None)
    vec = eigenvectors[0]

    angle_deg = np.rad2deg(np.arctan2(vec[1], vec[0]))
    corrected_angle = -angle_deg + 90 + PCA_ANGLE_OFFSET

    if debug:
        print(f"PCA corrected_angle: {corrected_angle}")
        debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        center = tuple(np.mean(coords, axis=0).astype(int))
        vec_draw = eigenvectors[0] * 100
        p1 = (int(center[1]), int(center[0]))
        p2 = (int(center[1] + vec_draw[1]), int(center[0] + vec_draw[0]))
        cv2.arrowedLine(debug_img, p1, p2, (0, 255, 0), 2)
        cv2.imshow("PCA Orientation", debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if corrected_angle < -89 or corrected_angle > 89:
        return image, corrected_angle

    rotated = rotate_image(image, corrected_angle)
    return rotated, corrected_angle


def rotate_image(image, angle, debug=DEBUG_MODE):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    border_value = 255 if len(image.shape) == 2 else (255,) * image.shape[2]
    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)

    if debug:
        cv2.imshow("rotated", rotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return rotated


def pad_image(img, height_ratio):
    h, w = img.shape[:2]
    target_h = int(w * height_ratio)
    if h < target_h:
        return img
    crop_top = (h - target_h) // 2
    crop_bottom = h - target_h - crop_top
    return img[crop_top:h - crop_bottom, :]


def crop_to_black_content(img, min_black_pixels=MIN_BLACK_PIXELS, margin=TO_BLACK_MARGIN, debug=DEBUG_MODE):
    """
    Przycina obraz na podstawie liczby czarnych pikseli w wierszach i kolumnach.
    """
    # Konwersja na skale szarości jeśli potrzebna
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Policz ilość czarnych pikseli w każdej kolumnie i wierszu
    black_per_row = np.sum(gray == 0, axis=1)
    black_per_col = np.sum(gray == 0, axis=0)

    # Szukamy pierwszego i ostatniego wiersza z wystarczającą liczbą czarnych pikseli
    rows = np.where(black_per_row >= min_black_pixels)[0]
    cols = np.where(black_per_col >= min_black_pixels)[0]

    # Jeżeli nie znaleziono wystarczająco czarnych pikseli - zwróć oryginalny obraz
    if len(rows) == 0 or len(cols) == 0:
        if debug:
            print("⚠️ Brak wystarczającej liczby czarnych pikseli - zwracam oryginał")
        return img

    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]

    if debug:
        print(f"before margin: top, bottom, left, right {top}, {bottom}, {left}, {right}")

    # Dodaj marginesy i upewnij się, że nie wychodzimy poza obraz
    h, w = gray.shape[:2]
    top = max(top - margin, 0)
    bottom = min(bottom + margin, h - 1)
    left = max(left - margin, 0)
    right = min(right + margin, w - 1)

    # Przycinamy obraz
    cropped = safe_crop(img, top, bottom, left, right)

    if debug:
        print(f"Przycinam do: top={top}, bottom={bottom}, left={left}, right={right}")
        cv2.imshow("Cropped", cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cropped


def safe_crop(img, top, bottom, left, right):
    h, w = img.shape[:2]
    top = max(0, min(top, h-1))
    bottom = max(top+1, min(bottom, h-1))
    left = max(0, min(left, w-1))
    right = max(left+1, min(right, w-1))
    return img[top:bottom+1, left:right+1]


def center_image(img, target_size=CENTER_TARGET_SIZE, debug=DEBUG_MODE):
    img = resize_image(img, target_size)
    background = np.ones((target_size, target_size), dtype=np.uint8) * 255
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    new_h, new_w = img.shape[:2]
    if new_h > target_size or new_w > target_size:
        raise ValueError("Obraz większy niż docelowe tło.")

    centroid = compute_centroid(gray)
    if centroid is None:
        start_y = (target_size - new_h) // 2
        start_x = (target_size - new_w) // 2
    else:
        centroid_y, centroid_x = centroid
        desired_center = target_size // 2
        start_y = max(0, min(desired_center - centroid_y, target_size - new_h))
        start_x = max(0, min(desired_center - centroid_x, target_size - new_w))

    background[start_y:start_y + new_h, start_x:start_x + new_w] = img
    return background


def divide_image(image, size=CENTER_TARGET_SIZE, debug=DEBUG_MODE):
    # image = resize_image(image, size)

    black_pixel_counts = np.sum(image == 0, axis=1)
    first = np.argmax(black_pixel_counts > DIVIDE_IMAGE_THRESHOLD)
    last = len(black_pixel_counts) - 1 - np.argmax(black_pixel_counts[::-1] > DIVIDE_IMAGE_THRESHOLD)
    counts_cropped = black_pixel_counts[first:last + 1]

    start, end = find_longest_sequence_below_threshold(counts_cropped, int(DIVIDE_IMAGE_THRESHOLD/3))
    center_row = (start + end) // 2
    image_cropped = image[first-TO_BLACK_MARGIN:last + 1 + TO_BLACK_MARGIN]
    center_row = center_row + TO_BLACK_MARGIN

    if (image_cropped[:center_row].shape[0] == 0 or image_cropped[center_row:].shape[0] == 0
        or image_cropped[:center_row].shape[1] == 0 or image_cropped[center_row:].shape[1] == 0):
        print(f"Nie udało się podzielić obrazu!")
        return image.copy(), image.copy()

    return image_cropped[:center_row], image_cropped[center_row:]


def resize_image(image, size, debug=DEBUG_MODE):
    new_size = (size, int((size * image.shape[0]) // image.shape[1]))
    if debug:
        print(f"target size: {size}")
        print(f"image.shape[0],[1]: {image.shape[0]},{image.shape[1]}")
        print(f"new_size: {new_size}")
    image = Image.fromarray(image)
    image = image.resize(new_size, Image.BICUBIC)
    image = np.array(image)

    return image

def find_longest_sequence_below_threshold(arr, threshold):
    max_len = curr_len = max_start = curr_start = 0
    for i, val in enumerate(arr):
        if val < threshold:
            if curr_len == 0:
                curr_start = i
            curr_len += 1
            if curr_len > max_len:
                max_len, max_start = curr_len, curr_start
        else:
            curr_len = 0
    return max_start, max_start + max_len - 1


def calculate_threshold_value_based_on_green(image_path, offset_down=OFFSET_DOWN, debug=DEBUG_MODE):
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Plik nie istnieje: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Nie można wczytać obrazu: {image_path}")

    h, w = image.shape[:2]
    roi = image[h - RECT_OFFSET - RECT_SIZE:h - RECT_OFFSET, RECT_OFFSET:RECT_OFFSET + RECT_SIZE]
    mean_green = np.mean(roi[:, :, 1])
    threshold_value = int(np.clip((GREEN_COEF_A * mean_green) + GREEN_COEF_B + offset_down, 0, 255))

    if debug:
        print(f"Obliczony próg: {threshold_value}")
    return threshold_value


def adaptive_hist_threshold(img,
                            threshold_estimate,
                            threshold_range=THRESHOLD_RAGE,
                            window_size=THRESHOLD_WINDOW_SIZE,
                            green_channel_only=GREEN_CHANNEL_ONLY,
                            debug=DEBUG_MODE):
    # Wstępna konwersja na odcień szarości (lub zielony kanał)
    if green_channel_only:
        if len(img.shape) == 3:
            channel = img[:, :, 1]  # zielony kanał
        else:
            channel = img.copy()
    else:
        if len(img.shape) == 3:
            channel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            channel = img.copy()

    # Konwersja do float32
    channel = channel.astype(np.float32)

    # Wyliczamy lokalną średnią jasności w oknie
    local_mean = cv2.boxFilter(channel, ddepth=-1, ksize=(window_size, window_size), normalize=True)

    # Ograniczamy próg w zakresie estimate ± range
    lower_bound = threshold_estimate - threshold_range
    upper_bound = threshold_estimate + threshold_range
    threshold_map = np.clip(local_mean, lower_bound, upper_bound).astype(np.uint8)

    # Właściwe progowanie adaptacyjne
    thresholded = (channel < threshold_map).astype(np.uint8) * 255

    thresholded = cv2.bitwise_not(thresholded)

    if debug:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.title("Kanał wejściowy")
        plt.imshow(channel, cmap='gray')
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.title(f"Mapa progów (range ±{threshold_range})")
        plt.imshow(threshold_map, cmap='gray')
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.title("Wynik progowania")
        plt.imshow(thresholded, cmap='gray')

        plt.tight_layout()
        plt.show()

    return thresholded.astype(np.uint8)

