import torch
import numpy as np
import cv2
from PIL import Image
import os

# === Importy stałych ===
from src.config import CTCOCR_MODEL_PATH, CTCOCR_RESIZE_TO, TROCR_DEVICE

# === Importy do modelu ===
from src.models.ctc_model.date.ctc_model import CTCModel
from src.models.ctc_model.date.ctc_alphabet_date import date_alphabet, date_char_to_idx, date_idx_to_char
from src.models.ctc_model.code.ctc_alphabet_code import code_alphabet, code_char_to_idx, code_idx_to_char

# === Parametry ===
class CTCDecoder:
    def __init__(self,
                 model_path=CTCOCR_MODEL_PATH,
                 text_type="date",
                 device=TROCR_DEVICE,
                 input_size=CTCOCR_RESIZE_TO):
        self.device = self.get_torch_device(device)
        self.input_size = input_size

        if text_type == "date":
            self.model = CTCModel(num_classes=len(date_alphabet) + 1).to(self.device)
        else:
            self.model = CTCModel(num_classes=len(code_alphabet) + 1).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        torch.set_grad_enabled(False)

    def get_torch_device(self, device_string):
        if device_string.lower() == "gpu":
            if torch.cuda.is_available():
                print("🟢 Wykryto GPU: używamy CUDA")
                return torch.device("cuda")
            else:
                print("⚠️ GPU wybrane, ale niedostępne → CPU")
                return torch.device("cpu")
        else:
            print("🟡 Używamy CPU")
            return torch.device("cpu")

    def preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            if image.ndim == 3:  # BGR albo BGRA
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif isinstance(image, Image.Image):
            image = image.convert("L")
            image = np.array(image)
        else:
            raise ValueError("Obraz musi być PIL.Image.Image lub NumPy ndarray.")

        # 2) Wymiary -> (H, W, 1)
        image = image[..., None]

        # 3) Resize 400×400
        target_dim = (400, 120)  # (szer, wys_crop)
        image = cv2.resize(image, (target_dim[0], target_dim[0]), interpolation=cv2.INTER_AREA)

        # 4) Center-crop do 120 px wysokości
        h = image.shape[0]
        top = max((h - target_dim[1]) // 2, 0)
        image = image[top: top + target_dim[1], :]

        if image.ndim == 2:  # brak kanału → przywróć
            image = image[..., None]  # → (H, W, 1)

        # 5) Skalowanie 0-1 + NORMALIZACJA takie jak w Albumentations
        image = image.astype(np.float32) / 255.0
        image = (image - 0.9661) / 0.1657   # Normalizacja po wartościach z trenowania

        # 6) [C, H, W] + batch
        image = np.transpose(image, (2, 0, 1))  # (1, H, W)
        image = torch.tensor(image).unsqueeze(0).float().to(self.device)
        return image

    def decode_logits(self, logits, text_type):
        logits = logits.cpu().detach().numpy()
        argmax = logits.argmax(axis=2)
        decoded = ""
        prev = -1
        for idx in argmax[0]:
            if text_type == "date":
                if idx != prev and idx != len(date_alphabet):
                    decoded += date_idx_to_char[idx]
            else:
                if idx != prev and idx != len(code_alphabet):
                    decoded += code_idx_to_char[idx]
            prev = idx
        return decoded

    def decode_ufi(self, image, text_type, return_times=False):
        import time
        total_start = time.perf_counter()

        image_tensor = self.preprocess_image(image)
        preproc_done = time.perf_counter()

        with torch.no_grad():
            decode_start = time.perf_counter()
            logits = self.model(image_tensor).permute(1, 0, 2)  # [B, T, C]
            prediction = self.decode_logits(logits, text_type)
            decode_time = time.perf_counter() - decode_start

        total_time = time.perf_counter() - total_start

        if return_times:
            return prediction, preproc_done - total_start, decode_time, total_time
        else:
            return prediction


if __name__ == "__main__":
    model_path = r"C:\Users\krukw\PycharmProjects\Baumer_test\src\models\ctc_trainedv1\best_ctc_model.pth"
    decoder = CTCDecoder(model_path=model_path, device="cpu")

    img_path = "C:/Users/krukw/PycharmProjects/Baumer_test/test_images/crop_test3.jpg"
    image = cv2.imread(img_path)

    decoded_msg, preproc_time, decode_time, total_time = decoder.decode_ufi(image, return_times=True)

    print("Odczytany tekst:", decoded_msg)
    print(f"Czas preprocessingu: {preproc_time:.3f} s")
    print(f"Czas dekodowania: {decode_time:.3f} s")
    print(f"Czas całkowity dekodowania: {total_time:.3f} s")

    from src.models.ctc_trainedv1.debug_ctc_input import debug_ctc_input
    debug_ctc_input(img_path, decoder)

