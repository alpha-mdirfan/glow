import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#  ----- Reading an image
# img = cv2.imread('opencv-icon.png', 1)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#  ---- Write an image
# img = cv2.imread('opencv-icon.png', 0)
# cv2.imshow('image', img)
# key = cv2.waitKey(0)
# if key == ord('s'):
#     cv2.imwrite("opencv_logo_GS.png", img)
# cv2.destroyAllWindows()

# --- Using MatplotLib
# img = cv2.imread('opencv-icon.png', 0)
# plt.imshow(img)
# plt.show()

# Face Detection with Eyes  -----------
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# img = cv2.imread('mee.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# for (x,y,w,h) in faces:
#     img = cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0),2)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = img[y:y+h, x:x+w]
#     eyes = eye_cascade.detectMultiScale(roi_gray)
#     for (ex, ey, ew, eh) in eyes:
#         cv2.rectangle(roi_color, (ex,ey), (ex+ey, ey+eh), (0,255,0), 2)

# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ----- Feature Detection ----------
img = cv.imread('Home.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
sift = cv.SIFT.create()
kp = sift.detect(gray, None)
img = cv.drawKeypoints(gray, kp, img)
cv.imwrite('keypoints.jpg', img)
cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()
