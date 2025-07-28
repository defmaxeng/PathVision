import cv2
import numpy as np



def show_bigger(img, scale=4.0):
    h, w = img.shape[:2]
    bigger = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imshow("Amplified", bigger)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


