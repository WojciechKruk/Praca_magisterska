import cv2
import pathlib
pathlib.PosixPath = pathlib.WindowsPath


def debug_crop(debug_images, debug_names, angle, angle2):
    debug_images = debug_images[::-1]
    debug_names = debug_names[::-1]
    i = 0
    for img in debug_images:
        print(f"wyświetlam {debug_names[i]}")
        cv2.imshow(debug_names[i], img)
        i = i + 1
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"angle 1: {angle}")
    print(f"angle 2: {angle2}")


def plot_one_box(xyxy, img, color=(0, 255, 0), label=None, line_thickness=2):
    # print("xyxy =", xyxy)
    # for i in range(len(xyxy)):
    #     print(f"xyxy[{i}] = {xyxy[i]}, type: {type(xyxy[i])}, shape: {getattr(xyxy[i], 'shape', 'no shape')}")
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(img, c1, c2, color, thickness=line_thickness)
    if label:
        font_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        t_size = cv2.getTextSize(label, font, font_scale, thickness=line_thickness)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] - 2), font, font_scale, [225, 255, 255], thickness=line_thickness)


if __name__ == "__main__":
    input("Naciśnij Enter, aby zakończyć...")

