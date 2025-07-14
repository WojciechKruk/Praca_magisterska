import sys
sys.path.append('/home/baumer/Baumer_neoAPI_1.5.0_lin_aarch64_python/wheel')
import neoapi

try:
    camera = neoapi.Cam()
    camera.Connect()
    print("Połączono z kamerą")

    if not camera.IsConnected():
        print("Kamera nie jest połączona")
    else:
        print(f"Rozdzielczość: {camera.f.Width.value} x {camera.f.Height.value}")
        camera.Disconnect()
        print("Kamera rozłączona")
except Exception as e:
    print("Wystąpił wyjątek:", e)


# import cv2
# import sys
# sys.path.append('/home/baumer/Baumer_neoAPI_1.5.0_lin_aarch64_python/wheel')
# import neoapi
# import time
# import os
#
# # Połącz z kamerą
# camera = neoapi.Cam()
# camera.Connect()
# print("Połączono z kamerą")
#
# if not camera.IsConnected():
#     print("Kamera nie jest połączona")
#     exit(1)
#
# # Ustawienie szerokości i wysokości obrazu
# if camera.f.Width.IsWritable() and camera.f.Height.IsWritable():
#     camera.f.Width.value = 1024
#     camera.f.Height.value = 768
#     print(f"Ustawiona rozdzielczość kamery: {camera.f.Width.value} x {camera.f.Height.value}")
#
# # Inicjalizacja zapisu wideo
# output_filename = "/home/baumer/nagranie_testowe.mp4"
#
# # Sprawdzamy, czy katalog istnieje
# if not os.path.exists(os.path.dirname(output_filename)):
#     print(f"Nie znaleziono katalogu {os.path.dirname(output_filename)}!")
#     exit(1)
#
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# fps = 15  # Ustaliliśmy FPS na 15
# frame_size = (camera.f.Width.value, camera.f.Height.value)
# video_writer = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)
#
# # Zapisz przez 5 sekund
# start_time = time.time()
# frame_count = 0
# while time.time() - start_time < 5:  # Nagrywaj przez 5 sekund
#     img = camera.GetImage()
#     if not img.IsEmpty():
#         imgarray = img.GetNPArray()
#
#         # Debug: Sprawdzenie kształtu obrazu
#         print(f"Zebrano klatkę {frame_count + 1} o rozmiarze: {imgarray.shape}")
#
#         video_writer.write(imgarray)  # Zapisz klatkę do pliku
#         print("Zapisano klatkę")
#         frame_count += 1
#     else:
#         print("Brak obrazu z kamery!")
#
# # Zakończ nagrywanie
# video_writer.release()
# print(f"Nagranie zapisane: {output_filename} ({frame_count} klatek)")
#
# # Zwolnienie zasobów
# camera.Disconnect()
# print("Kamera rozłączona.")
