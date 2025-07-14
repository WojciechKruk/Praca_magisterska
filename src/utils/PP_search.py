import cv2
import time
import numpy as np
from PIL import Image

# import stałych
from src.config import (TEMPLATE_PATH, LABEL_CENTER, LABEL_CENTER_PROP,
                        FLANN_INDEX_KDTREE, FLANN_TREES, FLANN_CHECKS,
                        ANGLE_OFFSET, DEBUG_MODE)


def find_angle(image_path, debug=DEBUG_MODE):
    template_path = str(TEMPLATE_PATH)

    angle = 0
    start = time.perf_counter()
    image = cv2.imread(image_path)
    if image is None:
        print(f"Błąd: Nie można wczytać obrazu {image_path}")
        return

    # reference_point = LABEL_CENTER
    reference_point = ((image.shape[1])*LABEL_CENTER_PROP[0], image.shape[0]*LABEL_CENTER_PROP[1])

    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"Błąd: Nie można wczytać pliku wzorca {template_path}")
        return

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(template, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image, None)

    if descriptors1 is not None and descriptors2 is not None:
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=FLANN_TREES)
        search_params = dict(checks=FLANN_CHECKS)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 10:
            if debug:
                print(f"Wykryto wystarczającą liczbę dopasowań")

            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = template.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)

            if debug:
                image = cv2.polylines(image, [np.int32(dst)], True, (0, 255, 0), 2)

            # Obliczamy środek wykrytego prostokąta
            rect_center = np.mean(dst.reshape(-1, 2), axis=0)
            cx, cy = rect_center

            # Obliczamy kąt względem poziomu (oś X)
            dx = cx - reference_point[0]
            dy = cy - reference_point[1]

            angle = np.degrees(np.arctan2(dy, dx))

            if debug:
                image = cv2.circle(image, (int(cx), int(cy)), 5, (255, 0, 0), -1)
                image = cv2.circle(image, (int(reference_point[0]), int(reference_point[1])), 5, (0, 0, 255), -1)
                image = cv2.line(image, (int(reference_point[0]), int(reference_point[1])), (int(cx), int(cy)), (0, 0, 255), 2)
                cv2.putText(image, f"Angle: {angle:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    angle = angle - ANGLE_OFFSET

    if debug:
        h, w = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D(reference_point, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

        newsize = (int(image.shape[0]*0.5), int(image.shape[1]*0.28))
        image = cv2.resize(image, newsize, interpolation=cv2.INTER_CUBIC)
        rotated_image = cv2.resize(rotated_image, newsize, interpolation=cv2.INTER_CUBIC)

        cv2.imshow("debug", image)
        cv2.imshow("rotated_image", rotated_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    t1 = time.perf_counter()
    return angle, float(t1 - start)


if __name__ == "__main__":
    find_angle(
        image_path="C:/Users/krukw/PycharmProjects/Baumer_test/test_images/test_image2.jpg"
    )
