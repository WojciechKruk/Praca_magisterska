import random
from PIL import Image
import numpy as np
import cv2
import math
from PIL import ImageDraw
from noise import pnoise2
import os

# === PERLIN NOISE SETTINGS ===
PERLIN_SCALE = 300.0  # im mniejsze, tym "bardziej zbite" szumy    *DEFAULT = 100.0
PERLIN_OCTAVES = 4  # liczba poziomów szczegółów                 *DEFAULT = 4
PERLIN_PERSISTENCE = 0.1  # jak bardzo kolejne oktawy wplywają         *DEFAULT = 0.5
PERLIN_LACUNARITY = 2.0  # jak szybko zmniejszają się detale          *DEFAULT = 2.0
PERLIN_STRENGTH = 100  # intensywnosc nałożone na obraz (0-255)     *DEFAULT = 60
PERLIN_SEED = None  # możesz podac int żeby mieć ten sam wzór    *DEFAULT = None


def random_date():
    day = f"{random.randint(1, 30):02}"
    month = f"{random.randint(1, 12):02}"
    year = f"{random.randint(2000, 2025)}"
    return f"{day}.{month}.{year}"


def random_code():
    letters = ''.join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=2))
    digit = str(random.randint(0, 9))
    hour = f"{random.randint(0, 23):02}:{random.randint(0, 59):02}"
    digits = ''.join(random.choices("0123456789", k=3))
    last_digit = str(random.randint(0, 9))
    return f"{letters}{digit} {hour} {digits} {last_digit}"


def apply_perlin_noise_overlay(image):
    image = image.convert("RGBA")
    width, height = image.size
    base = np.array(image).astype(np.float32)

    # Oddziel kanały
    rgb = base[:, :, :3]
    alpha = base[:, :, 3]

    # Generuj mapę szumu Perlin'a
    noise_map = np.zeros((height, width), dtype=np.float32)
    seed = PERLIN_SEED if PERLIN_SEED is not None else np.random.randint(0, 10000)

    for y in range(height):
        for x in range(width):
            nx = x / PERLIN_SCALE
            ny = y / PERLIN_SCALE
            noise = pnoise2(
                nx, ny,
                octaves=PERLIN_OCTAVES,
                persistence=PERLIN_PERSISTENCE,
                lacunarity=PERLIN_LACUNARITY,
                repeatx=width,
                repeaty=height,
                base=seed
            )
            noise_map[y, x] = (noise + 1) / 2  # skaluj z [-1,1] → [0,1]

    # Skaluj szum do RGB
    noise_rgb = (noise_map[:, :, None] * PERLIN_STRENGTH).repeat(3, axis=2)

    # Nakładanie szumu tylko tam, gdzie piksel nie jest w pełni przezroczysty
    mask = (alpha > 0).astype(np.float32)[..., None]
    rgb_noised = rgb + noise_rgb * mask

    # Sklej RGB i kanał alfa
    result = np.clip(rgb_noised, 0, 255).astype(np.uint8)
    output = np.dstack([result, alpha.astype(np.uint8)])

    return Image.fromarray(output, mode="RGBA")


def apply_texture_overlay(target_img, texture, texture_strength, pattern_scale):
    # 1. Skalujemy teksturę (zmniejszamy wzór)
    orig_w, orig_h = texture.size
    new_w = max(1, int(orig_w * pattern_scale))
    new_h = max(1, int(orig_h * pattern_scale))
    small_texture = texture.resize((new_w, new_h), Image.BICUBIC)

    # 2. Kafelkowanie tekstury z odbiciami lustrzanymi
    target_w, target_h = target_img.size
    reps_x = (target_w + new_w - 1) // new_w  # liczba powtórzeń poziomo
    reps_y = (target_h + new_h - 1) // new_h  # liczba powtórzeń pionowo

    tiled_texture = Image.new("L", (reps_x * new_w, reps_y * new_h))
    for i in range(reps_x):
        for j in range(reps_y):
            tile = small_texture
            if i % 2 == 1:
                tile = tile.transpose(Image.FLIP_LEFT_RIGHT)  # odbicie poziome
            if j % 2 == 1:
                tile = tile.transpose(Image.FLIP_TOP_BOTTOM)  # odbicie pionowe
            tiled_texture.paste(tile, (i * new_w, j * new_h))

    # Przycinamy do rozmiaru docelowego
    tiled_texture = tiled_texture.crop((0, 0, target_w, target_h))

    # Konwersja do numpy i normalizacja do [-1, 1]
    texture_np = np.array(tiled_texture).astype(np.float32)
    texture_norm = (texture_np - 128) / 127.0

    target_np = np.array(target_img).astype(np.float32)
    rgb = target_np[:, :, :3]
    alpha = target_np[:, :, 3:4]

    # Nakładamy teksturę na kanały RGB
    rgb += texture_norm[..., None] * 255 * texture_strength

    # Składamy wynik
    result = np.concatenate([np.clip(rgb, 0, 255), alpha], axis=2).astype(np.uint8)
    return Image.fromarray(result, mode="RGBA")


def generate_perlin_texture(width, height, scale=1000):
    texture = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            texture[y][x] = pnoise2(x / scale, y / scale, octaves=6)

    texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX)
    texture = texture.astype(np.uint8)

    kernel = np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]])
    embossed = cv2.filter2D(texture, -1, kernel) + 128  # Środek 128 dla efektu relief

    # Konwersja do obrazu PIL w trybie "L"
    pil_texture = Image.fromarray(np.clip(embossed, 0, 255).astype(np.uint8), mode="L")

    return pil_texture


def simulate_lighting_variation(img, light_brightness, light_color_shift):
    img = img.convert("RGBA")
    arr = np.array(img).astype(np.float32)

    arr[:, :, :3] *= light_brightness

    for c in range(3):
        arr[:, :, c] += light_color_shift[c]

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def apply_gradient_shadow(image, light_gradient_direction, light_gradient_strength):
    image = image.convert("RGBA")
    width, height = image.size
    gradient = np.ones((height, width), dtype=np.float32)

    if light_gradient_direction == "left":
        gradient *= np.linspace(1, 1 - light_gradient_strength, width)[None, :]
    elif light_gradient_direction == "right":
        gradient *= np.linspace(1 - light_gradient_strength, 1, width)[None, :]
    elif light_gradient_direction == "top":
        gradient *= np.linspace(1, 1 - light_gradient_strength, height)[:, None]
    elif light_gradient_direction == "bottom":
        gradient *= np.linspace(1 - light_gradient_strength, 1, height)[:, None]

    base = np.array(image).astype(np.float32)
    base[:, :, :3] *= gradient[..., None]
    base = np.clip(base, 0, 255).astype(np.uint8)

    return Image.fromarray(base, mode="RGBA")


def apply_spotlight_effect(image, light_spot_center, light_spot_std, light_spot_strength):
    image = image.convert("RGBA")
    width, height = image.size

    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xv, yv = np.meshgrid(x, y)

    cx, cy = light_spot_center
    std = light_spot_std

    dist_sq = (xv - cx) ** 2 + (yv - cy) ** 2
    spot_mask = 1 - light_spot_strength * np.exp(-dist_sq / (2 * std ** 2))

    base = np.array(image).astype(np.float32)
    base[:, :, :3] *= spot_mask[..., None]
    base = np.clip(base, 0, 255).astype(np.uint8)

    return Image.fromarray(base, mode="RGBA")


def sinus_wave(img, amplitude=5, period=100):
    arr = np.array(img)
    rows, cols = arr.shape[:2]
    for i in range(rows):
        shift = int(amplitude * np.sin(2 * np.pi * i / period))
        arr[i] = np.roll(arr[i], shift, axis=0)
    return Image.fromarray(arr)


def calculate_text_cords_and_angle(
            cords,
            random_cords,
            angle_text,
            angle_lid,
            lid_center,
            text_new_size,
            y_center_offset,
            lid_centure,
            translated_lid,
            rotated_lid,
            debug=False):
    # koordynaty wklejenia nadruku
    text_paste_cords = (cords[0] + random_cords[0],
                        cords[1] + random_cords[1])

    # koordynaty środka wklejonego nadruku daty
    date_paste_center_cords = (text_paste_cords[0] + text_new_size[0]/2,
                               text_paste_cords[1] + text_new_size[1]/2)

    # koordynaty środka wklejonego nadruku tekstu
    text_paste_center_cords = (date_paste_center_cords[0],
                               date_paste_center_cords[1] + y_center_offset)

    angle_text = -angle_text

    # koordynaty środka wklejonego nadruku tekstu po obróceniu
    text_paste_center_rotated1_cords = rotate_point(text_paste_center_cords, date_paste_center_cords, angle_text)

    if debug:
        translated_lid = draw_point(translated_lid, lid_center, "center")
        translated_lid = draw_arrow(translated_lid, text_paste_cords, 0, "Text paste location")
        translated_lid = draw_arrow(translated_lid, text_paste_center_rotated1_cords, angle_text, "Text center before rotation")
        translated_lid.show()

    if debug:
        print(f"angle_lid: {angle_lid}")
    angle_lid = -angle_lid
    text_center_rotated2_cords = rotate_point(text_paste_center_rotated1_cords, lid_center, angle_lid)

    angle_rotated_text = angle_lid + angle_text

    if debug:
        rotated_lid = draw_point(rotated_lid, lid_center, "center")
        rotated_lid = draw_arrow(rotated_lid, text_center_rotated2_cords, angle_rotated_text, "Text center roteted location")
        rotated_lid.show()

    # translacja po wklejeniu wieczka
    text_cords_final = (int(text_center_rotated2_cords[0] + (lid_centure[0] - translated_lid.size[0] // 2)),
                        int(text_center_rotated2_cords[1] + (lid_centure[1] - translated_lid.size[1] // 2)))

    return text_cords_final, angle_rotated_text


def rotate_point(pt, center, ang_deg):
    x,  y  = pt
    cx, cy = center
    t      = math.radians(ang_deg)
    dx, dy = x - cx, y - cy
    x_r =  dx*math.cos(t) - dy*math.sin(t)
    y_r =  dx*math.sin(t) + dy*math.cos(t)
    return (cx + x_r, cy + y_r)


def draw_point(img, xy, label=None, color="green", r=6, label_offset=(6, -12)):
    draw = ImageDraw.Draw(img)
    x, y = xy
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    if label is not None:
        dx, dy = label_offset
        draw.text((x + dx, y + dy), label, fill=color)

    return img

def draw_arrow(img, pt, ang, label):
    draw = ImageDraw.Draw(img)
    draw.ellipse((pt[0]-4, pt[1]-4, pt[0]+4, pt[1]+4), fill="blue")
    draw.text((pt[0]+6, pt[1]-12), label, fill="blue")
    ang_r = math.radians(ang)
    end = (pt[0]+80*math.cos(ang_r),
            pt[1]+80*math.sin(ang_r))
    draw.line([pt, end], fill="red", width=3)
    head = 10
    left  = (end[0]-head*math.cos(ang_r-0.35),
                end[1]-head*math.sin(ang_r-0.35))
    right = (end[0]-head*math.cos(ang_r+0.35),
                end[1]-head*math.sin(ang_r+0.35))
    draw.polygon([end, left, right], fill="red")

    return img

def save_loc(text_cords, text_angle, OUTPUT_DIR, i):
    with open(os.path.join(OUTPUT_DIR, "loc_labels.txt"), "a") as f:
        if i < 10:
            f.write(f"img_000{i}.jpg {text_cords} {text_angle}\n")
        elif i < 100:
            f.write(f"img_00{i}.jpg {text_cords} {text_angle}\n")
        elif i < 1000:
            f.write(f"img_0{i}.jpg {text_cords} {text_angle}\n")
        else:
            f.write(f"img_{i}.jpg {text_cords} {text_angle}\n")

def text_loc_debug(img, text_cords, text_angle):
    debug_img = draw_arrow(img, text_cords, text_angle, "text_loc_debug")
    debug_img.show()