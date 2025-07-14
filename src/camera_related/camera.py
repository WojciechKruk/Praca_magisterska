#!/usr/bin/env python3

import sys
import cv2
import torch
import numpy as np
import neoapi

# Załaduj YOLOv5
model_path = "../../best.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"YOLO działa na: {device}")
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True).to(device)
model.conf = 0.5  # Pewność detekcji
model.max_det = 10  # Maksymalna liczba obiektów
print(f"PyTorch używa GPU: {torch.cuda.is_available()}")

try:
    # Połącz z kamerą
    camera = neoapi.Cam()
    camera.Connect()
    print("Połączono z kamerą")

    if not camera.IsConnected():
        print("Kamera nie jest połączona")
        sys.exit(1)

    # Ustawienie szerokości i wysokości tak, aby kamera wysyłała obraz 1024x768
    if camera.f.Width.IsWritable() and camera.f.Height.IsWritable():
        camera.f.Width.value = 1024
        camera.f.Height.value = 768

    print(f"Ustawiona rozdzielczość kamery: {camera.f.Width.value} x {camera.f.Height.value}")

    # Binning dla optymalizacji
    if camera.f.BinningHorizontal.IsWritable() and camera.f.BinningVertical.IsWritable():
        camera.f.BinningHorizontal.value = 3
        camera.f.BinningVertical.value = 3

    # Ustawienie formatu obrazu
    if camera.f.PixelFormat.IsWritable():
        if camera.f.PixelFormat.GetEnumValueList().IsReadable("RGB8"):
            camera.f.PixelFormat.value = neoapi.PixelFormat_RGB8
        elif camera.f.PixelFormat.GetEnumValueList().IsReadable("Mono8"):
            camera.f.PixelFormat.value = neoapi.PixelFormat_Mono8
        print(f"Ustawiono format obrazu: {camera.f.PixelFormat.value}")

    # Zmniejsz FPS dla lepszej wydajności
    if camera.f.AcquisitionFrameRate.IsWritable():
        camera.f.AcquisitionFrameRate.value = 15

    # Synchronizacja ustawień
    camera.SetSynchronFeatureMode(True)

    # Otwórz okno podglądu
    cv2.namedWindow("YOLOv5 Detekcja", cv2.WINDOW_NORMAL)

    # # **INICJALIZACJA VIDEO WRITERA**
    output_filename = "nagranie_wykrywanie.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Kodowanie MP4
    fps = 15  # Tyle FPS, ile ustaliliśmy w kamerze
    frame_size = (camera.f.Width.value, camera.f.Height.value)
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

    # print(f"🎥 Nagrywanie do pliku: {output_filename}")

    while True:
        img = camera.GetImage()
        if not img.IsEmpty():
            imgarray = img.GetNPArray()

            # Konwersja BGR -> RGB (dla YOLO)
            img_rgb = cv2.cvtColor(imgarray, cv2.COLOR_BGR2RGB)

            # Detekcja YOLO
            results = model(img_rgb)

            # Rysowanie detekcji na obrazie
            for det in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = det
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(imgarray, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(imgarray, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Wyświetlanie obrazu
            cv2.imshow("YOLOv5 Detekcja", imgarray)

            # # **Zapis klatki do pliku**
            video_writer.write(imgarray)

        # Wyjście po ESC
        if cv2.waitKey(10) == 27:
            break

    # # **ZAKOŃCZENIE NAGRYWANIA**
    video_writer.release()
    print(f"Nagranie zapisane: {output_filename}")

    # Zwolnienie zasobów
    cv2.destroyAllWindows()
    camera.Disconnect()
    print("Kamera rozłączona.")

except (neoapi.NeoException, Exception) as exc:
    print("Błąd:", exc)
    sys.exit(1)

sys.exit(0)
