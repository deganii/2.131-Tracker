import cv2
import numpy as np
import cv2 as cv

from PIL import Image

is_old_cv = False

def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return 0
  return w*h

def mser_ecoli():
    if is_old_cv:
        mser = cv.MSER()
    else:
        mser = cv.MSER_create( # cv.MSER_create()
            _delta=5,
            _min_area=720,
            _max_area=9000,
            _max_variation=15.0,
            _min_diversity=10.0,
            _max_evolution=10,
            _area_threshold=12.0,
            _min_margin=2.9,
            _edge_blur_size=10)

    # img_path = 'C:\\dev\\courses\\2.131 - Advanced Instrumentation\\E.coli.tif'
    img_path = 'C:\\dev\\Holographic-Images\\Combined-Half-and Half-capture-2018-04-30-20h-52m-09s.png'
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
    # cv.polylines(vis, hulls, 1, (0, 255, 0))


    #boundingboxes
    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    mask = cv2.dilate(mask, np.ones((150,150), np.uint8))

    for contour in hulls:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)


    bboxes = []
    for i, contour in enumerate(hulls):
        x,y,w,h = cv2.boundingRect(contour)
        bboxes.append(cv2.boundingRect(contour))
        #cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for i in range(len(bboxes)):
        j = i + 1
        while j < len(bboxes):
            if intersection(bboxes[i], bboxes[j]) > 0:
                break
            else:
                j = j + 1
        if j == len(bboxes):
            x, y, w, h = bboxes[i]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return img, vis, bboxes
    cv2.imshow('img', vis)
    cv2.waitKey()


def mser_ecoli2(img, vis, bboxes1):
    if is_old_cv:
        mser = cv.MSER()
    else:
        mser = cv.MSER_create( # cv.MSER_create()
            _delta=4,
            _min_area=500,
            _max_area=2000,
            _max_variation=15.0,
            _min_diversity=10.0,
            _max_evolution=10,
            _area_threshold=12.0,
            _min_margin=2.9,
            _edge_blur_size=10)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if is_old_cv:
        regions = mser.detect(gray, None)
    else:
        regions, q = mser.detectRegions(gray)

    #polylines
    hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    # cv.polylines(vis, hulls, 1, (0, 255, 0))


    #boundingboxes
    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    mask = cv2.dilate(mask, np.ones((150,150), np.uint8))

    for contour in hulls:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)


    bboxes = []
    for i, contour in enumerate(hulls):
        x,y,w,h = cv2.boundingRect(contour)
        bboxes.append(cv2.boundingRect(contour))
        #cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    bboxesAll = bboxes + bboxes1
    for i in range(len(bboxes)):
        j = i + 1
        while j < len(bboxesAll):
            if intersection(bboxes[i], bboxesAll[j]) > 0:
                break
            else:
                j = j + 1
        if j == len(bboxesAll):
            x, y, w, h = bboxes[i]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('img', vis)
    cv2.imwrite('classified.jpg', vis)
    cv2.waitKey()




img, vis, bboxes1 = mser_ecoli()
mser_ecoli2(img, vis, bboxes1)