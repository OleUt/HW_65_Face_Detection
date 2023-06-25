# HOG
# Не виконувалось в Jupyter notebook, тільки в PyCharm. Результати в окремих файлах.

import dlib
from imutils import face_utils
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('titanic.jpg')
plt.imshow(img)
# plt.show()

gray = cv2.imread('titanic.jpg', 0)
plt.imshow(gray)
# plt.show()
plt.savefig('hog_gray.jpg')

im = np.float32(gray) / 255.0
gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
plt.imshow(mag)
# plt.show()
plt.savefig('hog_contur.jpg')

face_detect = dlib.get_frontal_face_detector()
rects = face_detect(gray, 1)

for (i, rect) in enumerate(rects):
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(gray, (x, y), (x+w, y+h), (255,160,122), 3)

plt.figure(figsize=(10,6))
plt.imshow(gray, cmap='gray')
# plt.show()
plt.savefig('hog_frame.jpg')