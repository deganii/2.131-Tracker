import cv2
import numpy as np
import cv2 as cv

from PIL import Image

is_old_cv = False

def mser_ecoli():
    if is_old_cv:
        mser = cv.MSER()
    else:
        mser = cv.MSER_create()

    img_path = 'C:\\dev\\courses\\2.131 - Advanced Instrumentation\\E.coli.tif'
    pil_img = Image.open(img_path)

    # for i in range(149):
    #     pil_img.seek(i)
    img= np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    vis = img.copy()
    if is_old_cv:
        regions = mser.detect(gray, None)
    else:
        regions, q = mser.detectRegions(gray)

    #polylines
    hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv.polylines(vis, hulls, 1, (0, 255, 0))


    #boundingboxes
    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    mask = cv2.dilate(mask, np.ones((150,150), np.uint8))

    for contour in hulls:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    for i, contour in enumerate(hulls):
        x,y,w,h = cv2.boundingRect(contour)
        print(x,y,w,h) #print coordinates and dimensions of bounding boxes

    cv2.imshow('img', vis)
    cv2.waitKey()



mser_ecoli()