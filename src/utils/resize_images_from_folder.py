import os
from PIL import Image

# 🔧 Parametry do konfiguracji:
input_folder = r"C:\Users\krukw\PycharmProjects\Baumer_test\pom\in"
output_folder = r'C:\Users\krukw\PycharmProjects\Baumer_test\pom\out'
target_size = (1000, 1000)  # (szerokość, wysokość) (dla CV-X (2432, 1824))
max_images = 1200  # Ile obrazów maksymalnie przetworzyć

# 🗂️ Utwórz folder wyjściowy jeśli nie istnieje
os.makedirs(output_folder, exist_ok=True)

# 📷 Zbieranie listy obrazów
image_files = [f for f in os.listdir(input_folder)
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

# ✂️ Ogranicz liczbę obrazów
image_files = image_files[:max_images]

for filename in image_files:
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    try:
        with Image.open(input_path) as img:
            resized_img = img.resize(target_size, Image.BICUBIC)
            resized_img.save(output_path)
            print(f"Zapisano: {output_path}")
    except Exception as e:
        print(f"Błąd przetwarzania {filename}: {e}")
