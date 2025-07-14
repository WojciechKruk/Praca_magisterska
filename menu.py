from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QFileDialog, QMessageBox, QSpinBox, QPushButton, QHBoxLayout
)
from PySide6.QtGui import QPixmap, QImage, QFont, QColor
from PySide6.QtCore import Qt, QTimer, QObject, Signal, Slot
from src.my_utils.ui_utils import UIHelpers
import sys
import cv2
import os
import time
from src.detect_image import load_detection_model, detect_and_process
DETECT_MODEL, STRIDE, NAMES, PT = load_detection_model()
from src.decode import TrOCRDecoder
from src.decode_ctc import CTCDecoder
from src.config import OCR_MODEL

if OCR_MODEL == "TROCR":
    OCR_DECODER = TrOCRDecoder()
if OCR_MODEL == "CTCOCR":
    OCR_DECODER = CTCDecoder()

DEFAULT_IMAGE_FOLDER = "C:/Users/krukw/PycharmProjects/Baumer_test/"
DEFAULT_IMAGE_NAME = "test_image.bmp"


class Worker(QObject):
    finished = Signal(object, object, float, float)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        result_img, detect_time, crops_top, crops_bottom, crop_time = detect_and_process(self.image_path, DETECT_MODEL, STRIDE, NAMES, PT)
        self.finished.emit(result_img, crops_top[0], crops_bottom[0], detect_time, crop_time)

class MenuApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("System Wykrywania")
        self.setGeometry(100, 100, 1280, 720)

        self.central = QWidget()
        self.setCentralWidget(self.central)
        self.layout = QVBoxLayout(self.central)
        self.layout.setAlignment(Qt.AlignCenter)
        self.central.setStyleSheet("background-color: #E5E8EC;")

        self.menu_options = [
            ("Wykryj kod", self.detect_code_menu),
            ("Generuj dane", self.generate_data_menu),
            ("Przetwórz dane", self.process_data_menu)
        ]

        self.create_menu()

    def create_menu(self):
        UIHelpers.clear(self.layout)
        UIHelpers.add_title(self.layout, "Wybierz opcję")

        for text, handler in self.menu_options:
            btn = UIHelpers.create_button(text, handler)
            self.layout.addSpacing(20)
            self.layout.addWidget(btn, alignment=Qt.AlignCenter)
            self.layout.addSpacing(20)

    ### DETEKCJA
    def detect_code_menu(self):
        UIHelpers.clear(self.layout)
        UIHelpers.add_title(self.layout, "Konfiguracja wykrywania")

        self.source_group, self.source_var = UIHelpers.radio_group(
            self.layout, "Wybierz źródło danych", ["Plik", "Domyślny", "Folder"])

        self.rotate_group, self.rotate_var = UIHelpers.radio_group(
            self.layout, "Automatyczny obrót", ["Nie", "Tak"])

        start_btn = UIHelpers.create_button("Rozpocznij", self.start_detect)
        self.layout.addWidget(start_btn, alignment=Qt.AlignCenter)
        UIHelpers.add_back(self.layout, self.create_menu)

    def start_detect(self):
        selected_source = self.source_group.checkedId()

        if selected_source == 0:
            file_path = QFileDialog.getOpenFileName(self, "Wybierz obraz", "", "Obrazy (*.jpg *.png *.jpeg *.bmp)")[0]
            if not file_path:
                self.create_menu()
                return
        else:
            file_path = os.path.join(DEFAULT_IMAGE_FOLDER, DEFAULT_IMAGE_NAME)
            if not os.path.exists(file_path):
                QMessageBox.critical(self, "Błąd", "Domyślny plik nie istnieje.")
                self.create_menu()
                return

        # Pokazujemy "przetwarzanie"
        UIHelpers.clear(self.layout)
        self.processing_label = QLabel("⏳ Przetwarzanie...")
        self.processing_label.setFont(UIHelpers.title_font())
        self.processing_label.setStyleSheet("color: #262626;")
        self.layout.addWidget(self.processing_label, alignment=Qt.AlignCenter)
        self.repaint()

        # Tu dodajemy mały delay żeby GUI zdążyło narysować label
        QTimer.singleShot(100, lambda: self.process_image(file_path))

    def process_image(self, path):
        try:
            result_img, detect_time, crops_top, crops_bottom, crop_time = detect_and_process(path, DETECT_MODEL, STRIDE, NAMES, PT)
            self.result_crop_top = crops_top[0]
            self.result_crop_bottom = crops_bottom[0]
            self.show_result(result_img, self.result_crop_top, self.result_crop_bottom, detect_time, crop_time)

        except Exception as e:
            QMessageBox.critical(self, "Błąd", f"Wystąpił problem:\n{e}")
            self.create_menu()

    def show_result(self, result_img, crop_img_top, crop_img_bottom, detect_time, crop_time):
        UIHelpers.clear(self.layout)

        result_pix = self.cv_to_pixmap(result_img)
        crop_pix_top = self.cv_to_pixmap(crop_img_top)
        crop_pix_bottom = self.cv_to_pixmap(crop_img_bottom)

        img_row = QHBoxLayout()
        img_label = QLabel(); img_label.setPixmap(result_pix.scaled(600, 300, Qt.KeepAspectRatio))
        crop_label_top = QLabel(); crop_label_top.setPixmap(crop_pix_top.scaled(600, 300, Qt.KeepAspectRatio))
        crop_label_bottom = QLabel(); crop_label_bottom.setPixmap(crop_pix_bottom.scaled(600, 300, Qt.KeepAspectRatio))
        img_row.addWidget(img_label)
        img_row.addWidget(crop_label_top)
        img_row.addWidget(crop_label_bottom)
        self.layout.addLayout(img_row)

        UIHelpers.add_label(self.layout, f"⏱️ Czas detekcji: {detect_time:.2f} s")
        UIHelpers.add_label(self.layout, f"🧹 Czas ROI: {crop_time:.2f} s")

        decode_btn = UIHelpers.create_button("📖 Odczytaj tekst", self.decode_crop)
        self.layout.addWidget(decode_btn, alignment=Qt.AlignCenter)
        UIHelpers.add_back(self.layout, self.create_menu)

    def decode_crop(self):
        # UIHelpers.clear(self.layout)
        # self.processing_label = QLabel("Odczytywanie...")
        # self.processing_label.setFont(UIHelpers.title_font())
        # self.processing_label.setStyleSheet("color: #262626;")
        # self.layout.addWidget(self.processing_label, alignment=Qt.AlignCenter)
        # self.repaint()
        # time.sleep(100)

        decoded_msg, preproc_time, decode_time, decode_total_time = OCR_DECODER.decode_ufi(self.result_crop_top,
                                                                                            return_times=True)
        UIHelpers.add_label(self.layout, f"📖 Odczytany tekst: {decoded_msg}")
        UIHelpers.add_label(self.layout, f"⏱️ Czas ładowania modelu: {OCR_DECODER.model_load_time:.2f} s")
        UIHelpers.add_label(self.layout, f"⏱️ Czas preprocesowania obrazu: {preproc_time:.2f} s")
        UIHelpers.add_label(self.layout, f"⏱️ Czas dekodowania: {decode_time:.2f} s")
        UIHelpers.add_label(self.layout, f"⏱️ Całkowity czas dekodowania: {decode_total_time:.2f} s")

    def cv_to_pixmap(self, img):
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_img)

    def generate_data_menu(self):
        UIHelpers.clear(self.layout)
        UIHelpers.add_title(self.layout, "Generowanie danych")

        self.num_images = QSpinBox()
        self.num_images.setValue(10)
        self.num_images.setMaximum(10000)

        self.res_x = QSpinBox()
        self.res_x.setValue(256)
        self.res_y = QSpinBox()
        self.res_y.setValue(256)

        UIHelpers.labeled_spinbox(self.layout, "Liczba obrazów:", self.num_images)
        UIHelpers.labeled_resolution(self.layout, self.res_x, self.res_y)

        start_btn = UIHelpers.create_button("Rozpocznij", self.start_generate_data)
        self.layout.addWidget(start_btn, alignment=Qt.AlignCenter)
        UIHelpers.add_back(self.layout, self.create_menu)

    def start_generate_data(self):
        folder = QFileDialog.getExistingDirectory(self, "Wybierz folder")
        if folder:
            QMessageBox.information(self, "Info", f"Generuję {self.num_images.value()} obrazów")
        self.create_menu()

    def process_data_menu(self):
        UIHelpers.clear(self.layout)
        UIHelpers.add_title(self.layout, "Przetwarzanie danych")

        self.process_group, self.process_var = UIHelpers.radio_group(
            self.layout, "", ["Tylko data", "Tylko kod", "Data i kod"])

        start_btn = UIHelpers.create_button("Rozpocznij", self.start_process_data)
        self.layout.addWidget(start_btn, alignment=Qt.AlignCenter)
        UIHelpers.add_back(self.layout, self.create_menu)

    def start_process_data(self):
        QMessageBox.information(self, "Info", "Proces zakończony")
        self.create_menu()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MenuApp()
    window.show()
    sys.exit(app.exec())