import os
import random
from PIL import Image
import numpy as np
import cv2
from font_generator import render_inkjet_text
from fake_data_utils import (random_date, random_code, apply_perlin_noise_overlay, apply_texture_overlay,
                             generate_perlin_texture, simulate_lighting_variation, apply_gradient_shadow,
                             apply_spotlight_effect, sinus_wave,
                             calculate_text_cords_and_angle, save_loc, text_loc_debug)

# importy stałych
from src.config import (
    SYN_IMG_SIZE, SYN_NUM_IMAGES, SYN_FIRST_IMAGE_NUMBER, DEBUG_MODE,
    SYN_OUTPUT_DIR, SYN_LID_DIR, LABEL_CENTER,
    SYN_TEXTURE_STRENGTH, SYN_PATTERN_SCALE,
    SYN_TEXT_IMG_SIZE_SCALE, SYN_TEXT_OFFSET_MAX, SYN_TEXT_SCALE_FACTOR,
    SYN_BASE_COLOR, SYN_KSIZE, SYN_ROZMYCIE,
    SYN_WAVE_AMPLITUDE, SYN_WAVE_PERIOD, SAVE_LOC
)


# ============ USTAWIENIA ============

# === PODSTAWOWE ===
IMG_SIZE = SYN_IMG_SIZE
NUM_IMAGES = SYN_NUM_IMAGES
FIRST_IMAGE_NUMBER = SYN_FIRST_IMAGE_NUMBER

# === ŚCIEŻKI ===
OUTPUT_DIR = SYN_OUTPUT_DIR
LID_DIR = SYN_LID_DIR

# === WIECZKO ===
LID_CENTER = LABEL_CENTER

# === TEKSTURY ===
TEXTURE_STRENGTH = SYN_TEXTURE_STRENGTH  # zakres 0.0 - 1.0
PATTERN_SCALE = SYN_PATTERN_SCALE

# === TEKST ===
TEXT_IMG_SIZE_SCALE = SYN_TEXT_IMG_SIZE_SCALE
TEXT_OFFSET_MAX = SYN_TEXT_OFFSET_MAX

# === KOLORY ===
BASE_COLOR = SYN_BASE_COLOR

# === ROZMYCIE ===
KSIZE = SYN_KSIZE
ROZMYCIE = SYN_ROZMYCIE

os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_image(i, debug=DEBUG_MODE):
    # Tło
    if debug:
        print("Generowanie tła...")
    img = Image.new("RGB", IMG_SIZE, BASE_COLOR)

    # Generowanie losowych parametrów
    light_brightness = random.uniform(0.85, 1.15)
    light_color_shift = [random.uniform(-7, 7) for _ in range(3)]
    light_gradient_direction = random.choice(["left", "right", "top", "bottom"])
    light_gradient_strength = random.uniform(0.1, 0.4)
    light_spot_center = (random.uniform(0.3, 0.7), random.uniform(0.3, 0.7))
    light_spot_std = random.uniform(0.3, 0.6)
    light_spot_strength = random.uniform(0.3, 0.6)

    # Uchwyt
    if debug:
        print("Wklejanie uchwytu...")
        img.show()
    grip_path = os.path.join(LID_DIR, "uchwyt.png").replace("\\", "/")
    grip = Image.open(grip_path).convert("RGBA")
    # Symulowanie światła uchwytu
    if debug:
        print("Symulowanie światła uchwytu...")
    grip = simulate_lighting_variation(grip, light_brightness, light_color_shift)
    grip = apply_gradient_shadow(grip, light_gradient_direction, light_gradient_strength)
    grip = apply_spotlight_effect(grip, light_spot_center, light_spot_std, light_spot_strength)
    img.paste(grip, (0, 0), grip)

    # Wieczko
    if debug:
        print("Wklejanie wieczka...")
        img.show()
    lid_path = os.path.join(LID_DIR, "wieczko.png").replace("\\", "/")
    lid = Image.open(lid_path).convert("RGBA")
    result_lid = Image.new("RGBA", lid.size, (0, 0, 0, 0))

    translated_lid = Image.new("RGBA", lid.size, (0, 0, 0, 0))
    translated_lid.paste(lid, (int(lid.size[0] // 2 - LID_CENTER[0]), int(lid.size[1] // 2 - LID_CENTER[1])))

    # tekstura
    if debug:
        print("Nakładanie tekstury 1...")
        translated_lid.show()
    lid_texture_path = os.path.join(LID_DIR, "tekstura_wieczka.jpg").replace("\\", "/")
    texture1 = Image.open(lid_texture_path).convert("L")
    translated_lid = apply_texture_overlay(translated_lid, texture1, TEXTURE_STRENGTH, PATTERN_SCALE)
    if debug:
        print("Generowanie i nakładanie tekstury 2...")
        translated_lid.show()
    texture2 = generate_perlin_texture(IMG_SIZE[0], IMG_SIZE[1])
    translated_lid = apply_texture_overlay(translated_lid, texture2, texture_strength=0.04, pattern_scale=PATTERN_SCALE)
    if debug:
        print("Nakładanie tekstury foli...")
        texture2.show()
        translated_lid.show()
    # foil_texture_names = [f"Foil002_1K-PNG_AmbientOcclusion.png", f"Foil002_1K-PNG_Displacement.png",
    #                       f"Foil002_1K-PNG_Metalness.png", f"Foil002_1K-PNG_NormalDX.png",
    #                       f"Foil002_1K-PNG_NormalGL.png", f"Foil002_1K-PNG_Roughness.png"]
    foil_path = os.path.join(LID_DIR, "foil_texture").replace("\\", "/")
    foil_texture_names = [f"Foil002_1K-PNG_AmbientOcclusion.png", f"Foil002_1K-PNG_Displacement.png",
                          f"Foil002_1K-PNG_NormalDX.png", f"Foil002_1K-PNG_NormalGL.png",
                          f"Foil002_1K-PNG_Roughness.png"]
    for j in range(5):
        foil_texture_path = os.path.join(foil_path, foil_texture_names[j]).replace("\\", "/")
        foil_texture = Image.open(foil_texture_path).convert("L")
        if j == 2 or j == 3:
            translated_lid = apply_texture_overlay(translated_lid, foil_texture, texture_strength=0.2, pattern_scale=6)
        else:
            translated_lid = apply_texture_overlay(translated_lid, foil_texture, texture_strength=0.08, pattern_scale=6)

    # Tekst
    if debug:
        print("Generowanie tekstu...")
        translated_lid.show()
    line1 = random_date()
    line2 = random_code()

    text_color = tuple(random.randint(5, 10) for _ in range(3))

    if debug:
        print("Generowanie obrazu tekstu...")
    text_scale_factor = SYN_TEXT_SCALE_FACTOR
    text_img_size = (IMG_SIZE[0] * TEXT_IMG_SIZE_SCALE, IMG_SIZE[1] * TEXT_IMG_SIZE_SCALE)
    line1_text = render_inkjet_text(line1, "BIG", text_color, text_img_size, text_scale_factor)
    line2_text = render_inkjet_text(line2, "SMALL", text_color, text_img_size, text_scale_factor)

    if debug:
        print("Rozmycie tekstu...")
    text_np = np.array(line1_text)
    blurred_np = cv2.GaussianBlur(text_np, (33, 33), sigmaX=4)
    line1_text = Image.fromarray(blurred_np)
    text_np = np.array(line2_text)
    blurred_np = cv2.GaussianBlur(text_np, (33, 33), sigmaX=2)
    line2_text = Image.fromarray(blurred_np)

    text_new_size = (int(IMG_SIZE[0] * TEXT_IMG_SIZE_SCALE / text_scale_factor),
                int(IMG_SIZE[1] * TEXT_IMG_SIZE_SCALE / text_scale_factor))
    line1_text = line1_text.resize(text_new_size, Image.BICUBIC)
    line2_text = line2_text.resize(text_new_size, Image.BICUBIC)

    cords = (int((IMG_SIZE[0] / 2) - ((IMG_SIZE[0] * TEXT_IMG_SIZE_SCALE / text_scale_factor) / 2)),
             int((IMG_SIZE[1] / 2) - ((IMG_SIZE[1] * TEXT_IMG_SIZE_SCALE / text_scale_factor) / 2)))

    # Deformacja falą
    if debug:
        print("Deformacja falą...")
    line1_text = sinus_wave(line1_text, amplitude=SYN_WAVE_AMPLITUDE, period=SYN_WAVE_PERIOD)
    line2_text = sinus_wave(line2_text, amplitude=SYN_WAVE_AMPLITUDE, period=SYN_WAVE_PERIOD)

    if debug:
        print("Obrót...")
    angle_text = random.uniform(-5, 5)
    line1_text = line1_text.rotate(angle_text, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0, 0))
    line2_text = line2_text.rotate(angle_text, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0, 0))

    if debug:
        from PIL import ImageDraw
        w, h = line1_text.size
        draw = ImageDraw.Draw(line1_text)
        draw.rectangle([(0, 0), (w - 1, h - 1)], outline="red", width=1)
        draw.line([(0, 0), (w, h)], fill="red", width=1)
        draw.line([(0, h), (w, 0)], fill="red", width=1)
        w2, h2 = line2_text.size
        draw2 = ImageDraw.Draw(line2_text)
        draw2.rectangle([(0, 0), (w2 - 1, h2 - 1)], outline="red", width=1)
        draw2.line([(0, 0), (w2, h2)], fill="red", width=1)
        draw.line([(0, h2), (w2, 0)], fill="red", width=1)

    if debug:
        print("Wklejanie tekstu...")
        line1_text.show()
    random_cords = random.sample(range(-TEXT_OFFSET_MAX, TEXT_OFFSET_MAX), 2)
    translated_lid.paste(line1_text, (cords[0] + random_cords[0], cords[1] + random_cords[1]), line1_text)
    translated_lid.paste(line2_text, (int(cords[0] + random_cords[0] - 75),
                                      int(cords[1] + random_cords[1] + 50)), line2_text)
    if debug:
        translated_lid.show()
    # Obrót
    angle_lid = random.uniform(-180, 180)
    # if debug:
    #     angle = 0  # do debugowania
    rotated_lid = translated_lid.rotate(angle_lid, resample=Image.BICUBIC, expand=False)

    # Zapis lokalizacji nadruku
    if SAVE_LOC:
        lid_center = translated_lid.size[00] / 2, translated_lid.size[1] / 2
        y_center_offset = 25
        text_cords, text_angle = calculate_text_cords_and_angle(
            cords,
            random_cords,
            angle_text,
            angle_lid,
            lid_center,
            text_new_size,
            y_center_offset,
            LID_CENTER,
            translated_lid,
            rotated_lid
        )

        save_loc(text_cords, text_angle, OUTPUT_DIR, i)

    result_lid.paste(rotated_lid, (int(LID_CENTER[0] - lid.size[0] // 2),
                                   int(LID_CENTER[1] - lid.size[1] // 2)), rotated_lid)

    # Symulowanie światła wieczka
    if debug:
        print("Symulowanie światła wieczka...")
        result_lid.show()
    result_lid = simulate_lighting_variation(result_lid, light_brightness, light_color_shift)
    result_lid = apply_gradient_shadow(result_lid, light_gradient_direction, light_gradient_strength)
    result_lid = apply_spotlight_effect(result_lid, light_spot_center, light_spot_std, light_spot_strength)

    # Perlin noise
    if debug:
        print("Nakładanie szumu...")
        result_lid.show()
    result_lid = apply_perlin_noise_overlay(result_lid)

    # Wklejanie wieczka
    if debug:
        print("Nakładanie szumu...")
        result_lid.show()
    img.paste(result_lid, (0, 0), result_lid)

    # Rozmycie Gaussowskie
    if debug:
        print("Rozmywanie...")
        result_lid.show()
    img = np.array(img)
    img = cv2.GaussianBlur(img, KSIZE, sigmaX=ROZMYCIE)

    # Zapisywanie
    result = Image.fromarray(img)
    if SAVE_LOC and debug:
        text_loc_debug(result, text_cords, text_angle)
    if debug:
        result.show()
    if not debug:
        if i < 10:
            result.save(os.path.join(OUTPUT_DIR, f"img_000{i}.jpg"))
            with open(os.path.join(OUTPUT_DIR, "labels.txt"), "a") as f:
                f.write(f"img_000{i}.jpg {line1} {line2}\n")
        elif i < 100:
            result.save(os.path.join(OUTPUT_DIR, f"img_00{i}.jpg"))
            with open(os.path.join(OUTPUT_DIR, "labels.txt"), "a") as f:
                f.write(f"img_00{i}.jpg {line1} {line2}\n")
        elif i < 1000:
            result.save(os.path.join(OUTPUT_DIR, f"img_0{i}.jpg"))
            with open(os.path.join(OUTPUT_DIR, "labels.txt"), "a") as f:
                f.write(f"img_0{i}.jpg {line1} {line2}\n")
        else:
            result.save(os.path.join(OUTPUT_DIR, f"img_{i}.jpg"))
            with open(os.path.join(OUTPUT_DIR, "labels.txt"), "a") as f:
                f.write(f"img_{i}.jpg {line1} {line2}\n")

        print(f"Wygenerowano: {i-FIRST_IMAGE_NUMBER+1}/{NUM_IMAGES} obrazów")


try:
    for i in range(FIRST_IMAGE_NUMBER, FIRST_IMAGE_NUMBER+NUM_IMAGES):
        generate_image(i)
    print("Wygenerowano dane")
except KeyboardInterrupt:
    print("\n🛑 Program został przerwany przez użytkownika (KeyboardInterrupt). Zamykanie...")
