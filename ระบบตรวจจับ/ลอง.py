import cv2

img = cv2.imread("img/7.jpg")
img = cv2.resize(img,(600,600))

cv2.imshow("MyImage", img)
cv2.waitKey(0)