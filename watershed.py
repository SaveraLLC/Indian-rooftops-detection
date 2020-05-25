"""
Title: Rooftops detection in Google Maps Image using watershed algorithm
Author: Jitender Singh Virk (Virksaab)
Date created: 17 July, 2018
Last Modified: 26 July, 2018
"""

import numpy as np
import cv2
import os
from time import time

# GET ALL IMAGES PATH FROM FOLDERS
imagepaths = []
for dirpath, dirnames, filenames in os.walk('Satellite Images of different areas in delhi'):
    for filename in filenames:
        # print(filename)
        imagepaths.append(os.path.join(dirpath, filename))
        
# ITERATE OVER ALL IMAGES
for i, imgpath in enumerate(imagepaths):
    # print(i, imgpath)

    start = time()

    bgrimg = cv2.imread(imgpath)
    bgrimg = cv2.resize(bgrimg, (800, 600), interpolation=cv2.INTER_AREA)
    bgrimgo = np.copy(bgrimg)

    # GET IMAGE AND RESIZE
    gray = cv2.imread(imgpath, 0)
    gray = cv2.resize(gray, (800, 600), interpolation=cv2.INTER_AREA)

    # THRESHOLDING
    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # noise removal
    kernel = np.ones((3,3), np.uint8)
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)
    gradient = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=1)
    # sure_bg = cv2.bitwise_not(sure_bg)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform ,0.7*dist_transform.max(), 255, cv2.THRESH_TRUNC)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(bgrimg, markers)
    bgrimg[markers == -1] = [255,0,0]

    # stacked2d = np.hstack((gray, opening))
    stacked3d = np.hstack((bgrimg, bgrimgo))

    print("Time:", time() - start)

    # cv2.imwrite("outdir/output{}.png".format(i), stacked3d)

    # cv2.imshow("filters", stacked2d)
    cv2.imshow("finals", stacked3d)

    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()