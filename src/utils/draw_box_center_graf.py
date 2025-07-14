import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cv2
from PIL import Image
import os
import math
import random

# === Parametry tła ===
canvas_width = 2048
canvas_height = 1536

SCALE_FACTOR = 30
TEXT_OFFSET_MAX = 4
OUTPUT_DIR = r"C:\Users\krukw\PycharmProjects\Baumer_test\src\my_utils\bb_image_gif"

# === Parametry prostokątów (w pikselach) ===
rect1 = {'x': canvas_width/2-235, 'y': canvas_height/2, 'width': 470, 'height': 37}  # mniej więcej środek tła
rect2 = {'x': rect1['x'] + 35, 'y': rect1['y'] + 11 + 37, 'width': 240, 'height': 28}

# === Oblicz środek obrotu ===
center_x = rect1['x'] + rect1['width'] / 2
center_y = rect1['y'] + ((rect1['height'] + rect2['height'] + 11) / 2)
rotation_center = np.array([center_x, center_y])


# === Pomocnicze funkcje ===
def get_corners(x, y, w, h):
    return np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ])


def rotate_point(p, angle_deg, center):
    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    return rotation_matrix @ (p - center) + center

# === Funkcja do rysowania układu przy dowolnym kącie ===
def draw_debug2(angle_deg):
    corners1 = get_corners(rect1['x'], rect1['y'], rect1['width'], rect1['height'])
    corners2 = get_corners(rect2['x'], rect2['y'], rect2['width'], rect2['height'])
    all_corners = np.vstack([corners1, corners2])
    rotated = np.array([rotate_point(p, angle_deg, rotation_center) for p in all_corners])

    min_xy = rotated.min(axis=0)
    max_xy = rotated.max(axis=0)
    bbox_center = (min_xy + max_xy) / 2

    scaled_bbox_center = bbox_center

    # skalowanie
    scaled_bbox_center = rotation_center + SCALE_FACTOR * (bbox_center - rotation_center)

    # Bounding box
    bbox_rect = np.array([
        min_xy,
        [max_xy[0], min_xy[1]],
        max_xy,
        [min_xy[0], max_xy[1]],
        min_xy
    ])

    # Podział na dwa prostokąty po rotacji
    rotated1 = rotated[:4]
    rotated2 = rotated[4:]

    # Rysowanie
    plt.figure(figsize=(12, 9))
    plt.title(f"Debug: Obrót o {angle_deg}°")
    plt.xlim(0, canvas_width)
    plt.ylim(canvas_height, 0)
    plt.plot(rotation_center[0], rotation_center[1], 'ko', label='Środek obrotu')
    plt.plot(bbox_center[0], bbox_center[1], 'go', label='Środek BBOX')
    plt.plot(scaled_bbox_center[0], scaled_bbox_center[1], 'yo', label='Przeskalowany środek BBOX')
    plt.plot(bbox_rect[:, 0], bbox_rect[:, 1], 'r--', label='Bounding box')

    for corners in [rotated1, rotated2]:
        corners = np.vstack([corners, corners[0]])  # zamknięcie pętli
        plt.plot(corners[:, 0], corners[:, 1], 'b-')

    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.legend()
    plt.show()


# === Funkcja do rysowania układu przy dowolnym kącie ===
def draw_debug(angle_deg, image):
    img = image.copy()

    # Punkty prostokątów i obrót
    corners1 = get_corners(rect1['x'], rect1['y'], rect1['width'], rect1['height'])
    corners2 = get_corners(rect2['x'], rect2['y'], rect2['width'], rect2['height'])
    all_corners = np.vstack([corners1, corners2])
    rotated = np.array([rotate_point(p, angle_deg, rotation_center) for p in all_corners])

    # Środek bounding boxa
    min_xy = rotated.min(axis=0)
    max_xy = rotated.max(axis=0)
    bbox_center = (min_xy + max_xy) / 2

    # Skalowanie środka bboxa względem środka obrotu
    scaled_bbox_center = rotation_center + SCALE_FACTOR * (bbox_center - rotation_center)

    # Rysowanie bounding boxa
    bbox_rect = np.array([
        min_xy,
        [max_xy[0], min_xy[1]],
        max_xy,
        [min_xy[0], max_xy[1]],
    ], dtype=np.int32)

    cv2.polylines(img, [bbox_rect.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 255), thickness=2)

    # Rysowanie prostokątów
    rotated1 = rotated[:4].astype(np.int32)
    rotated2 = rotated[4:].astype(np.int32)
    cv2.polylines(img, [rotated1.reshape(-1, 1, 2)], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.polylines(img, [rotated2.reshape(-1, 1, 2)], isClosed=True, color=(255, 0, 0), thickness=2)

    # Środek obrotu (czarny), środek bbox (zielony), przeskalowany środek bbox (ciemnozielony)
    cv2.circle(img, tuple(rotation_center.astype(int)), 6, (0, 0, 0), -1)
    cv2.circle(img, tuple(bbox_center.astype(int)), 6, (0, 255, 0), -1)
    cv2.circle(img, tuple(scaled_bbox_center.astype(int)), 6, (0, 128, 128), -1)

    return img


# === Oblicz środki bounding boxów dla 0–359° ===
corners1 = get_corners(rect1['x'], rect1['y'], rect1['width'], rect1['height'])
corners2 = get_corners(rect2['x'], rect2['y'], rect2['width'], rect2['height'])
all_corners = np.vstack([corners1, corners2])

bbox_centers = []
scaled_bbox_centers = []
distances = []

blank_img = np.ones((1536, 2048, 3), dtype=np.uint8) * 255  # biały obraz RGB

for angle in range(360):
    rotated = np.array([rotate_point(p, angle, rotation_center) for p in all_corners])
    min_xy = rotated.min(axis=0)
    max_xy = rotated.max(axis=0)
    bbox_center = (min_xy + max_xy)/2

    # random_cords = random.sample(range(-TEXT_OFFSET_MAX, TEXT_OFFSET_MAX), 2)
    # bbox_center[0] = bbox_center[0] + random_cords[0]
    # bbox_center[1] = bbox_center[1] + random_cords[1]
    # bbox_center = rotation_center + 1.4 * (bbox_center - rotation_center)

    # skalowanie
    scaled_bbox_center = rotation_center + SCALE_FACTOR * (bbox_center - rotation_center)

    # distance = np.linalg.norm(bbox_center - rotation_center)

    distance = int(math.sqrt(abs(bbox_center[0] - rotation_center[0])*abs(bbox_center[0] - rotation_center[0]) +
                         abs(bbox_center[1] - rotation_center[1])*abs(bbox_center[1] - rotation_center[1])))

    bbox_centers.append(bbox_center)
    scaled_bbox_centers.append(scaled_bbox_center)
    distances.append(distance)

    debug_image = draw_debug(angle, blank_img)
    debug_image = Image.fromarray(debug_image)
    # debug_image.show()
    debug_image.save(os.path.join(OUTPUT_DIR, f"img_{angle}.jpg"))

bbox_centers = np.array(bbox_centers)
distances = np.array(distances)

# === Kolorowanie punktów na podstawie odległości ===
norm = (distances - distances.min()) / (distances.max() - distances.min())
cmap = get_cmap("coolwarm")
colors = cmap(norm)

# === Rysowanie trajektorii środków BBOX ===
plt.figure(figsize=(10, 8))
plt.title("Środki bounding boxów (kolor wg odległości)")
plt.xlim(0, canvas_width)
plt.ylim(canvas_height, 0)
plt.plot(rotation_center[0], rotation_center[1], 'ko', label='Środek obrotu')
for pt, col in zip(scaled_bbox_centers, colors):
    plt.plot(pt[0], pt[1], 'o', color=col)
plt.gca().set_aspect('equal')
plt.grid(True)
plt.legend()
plt.show()

# === Histogram odległości ===
plt.figure(figsize=(8, 4))
plt.hist(distances, bins=30, color='gray', edgecolor='black')
plt.title("Histogram odległości środków BBOX od środka obrotu")
plt.xlabel("Odległość")
plt.ylabel("Liczba przypadków")
plt.grid(True)
plt.show()

# === Rysuj obrazek debugowy z kątem (możesz zmieniać wartość poniżej) ===
draw_debug2(angle_deg=60)