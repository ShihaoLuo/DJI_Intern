import cv2 as cv

aruco_dict = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
for i in range(5):
    img = cv.aruco.drawMarker(aruco_dict, i, 1000)
    cv.imwrite(str(i)+'.jpg', img)
