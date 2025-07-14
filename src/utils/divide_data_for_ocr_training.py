import os
import shutil
import random

INPUT_DIR = r"C:\Users\krukw\PycharmProjects\Baumer_test\src\generate_data\generated_fake_data_processed\date"
OUTPUT_DIR = r"C:\Users\krukw\PycharmProjects\Baumer_test\datasets\ocr_train_fake_dataset_datev4"
TRAIN_RATIO = 0.80


def load_labels(label_path):
    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    label_dict = {}
    for line in lines:
        parts = line.strip().split(maxsplit=1)  # Działa dla tabów i spacji
        if len(parts) != 2:
            continue
        filename, label = parts
        label_dict[filename] = label
    return label_dict


def split_and_save_data(
    input_dir, output_dir, train_ratio=0.8, seed=42
):
    image_dir = input_dir
    label_path = os.path.join(input_dir, "labels.txt")

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Nie znaleziono folderu z obrazami: {image_dir}")
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"Nie znaleziono pliku labels.txt: {label_path}")

    label_dict = load_labels(label_path)
    print(f"Załadowano {len(label_dict)} etykiet z labels.txt")

    image_files = [f for f in label_dict.keys() if os.path.isfile(os.path.join(image_dir, f))]
    missing_files = [f for f in label_dict if not os.path.isfile(os.path.join(image_dir, f))]
    if missing_files:
        print(f"Ostrzeżenie: brak {len(missing_files)} plików wymienionych w labels.txt (np. {missing_files[:3]})")

    random.seed(seed)
    random.shuffle(image_files)

    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    for split_name, files in [("train", train_files), ("val", val_files)]:
        split_img_dir = os.path.join(output_dir, split_name, "images")
        os.makedirs(split_img_dir, exist_ok=True)
        split_label_path = os.path.join(output_dir, split_name, "labels.txt")

        with open(split_label_path, "w", encoding="utf-8") as label_file:
            for filename in files:
                src = os.path.join(image_dir, filename)
                dst = os.path.join(split_img_dir, filename)
                shutil.copyfile(src, dst)

                label_file.write(f"{filename}\t{label_dict[filename]}\n")

    print(f"Podział zakończony. Trening: {len(train_files)}, Walidacja: {len(val_files)}")


if __name__ == "__main__":
    split_and_save_data(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, train_ratio=TRAIN_RATIO)
