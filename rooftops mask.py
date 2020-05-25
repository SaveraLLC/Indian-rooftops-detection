"""
Title: Rooftops detection in Google Maps Image
Author: Jitender Singh Virk (Virksaab)
Date created: 26 July, 2018
Last Modified: 28 July, 2018
"""
import numpy as np
import cv2
import os
import time
#from shapely.geometry import MultiPolygon, Polygon


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

def sharpen(img):
    # SHARPEN
    kernel_sharp = np.array(([-2, -2, -2], [-2, 17, -2], [-2, -2, -2]), dtype='int')
    return cv2.filter2D(img, -1, kernel_sharp)

def mask_roofs(bgrimg):
    # GET IMAGE AND RESIZE
    gray = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2GRAY)

    # SHARPEN
    sgray = sharpen(gray)
    sbgrimg = sharpen(bgrimg)

    # THRESHOLDING
    ret, mask = cv2.threshold(sgray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # EDGES
    edges = auto_canny(mask)
    invedges = cv2.bitwise_not(edges)

    # REFINE MASK
    mieg = cv2.bitwise_and(mask, invedges)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    refined = cv2.bitwise_and(mieg, opening)
    # refined = cv2.bitwise_not(refined)

    # CONVERT MASK TO MATCH WITH ORIGNAL IMAGE DIMENSIONS
    refined3d = sbgrimg.copy()
    vidx, hidx = refined.nonzero()
    for ii in range(len(vidx)):
        refined3d[vidx[ii]][hidx[ii]][0] = 0
        refined3d[vidx[ii]][hidx[ii]][1] = 255
        refined3d[vidx[ii]][hidx[ii]][2] = 255

    #DRAW CONTOURS
    # im2, contours, hierarchy = cv2.findContours(refined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for contour in contours:
    #     area = cv2.contourArea(contour)
 
    #     if area > 2000:
    #         # cv2.drawContours(bgrimg, contour, -1, (0, 255, 0), 2)
    #         cv2.fillPoly(bgrimg, pts =[contour], color=(0,255,255))
    #         # bgrimg = cv2.polylines(bgrimg,[contour],True,(0,255,255))

    return refined3d

# GET ALL IMAGES PATH FROM FOLDERS
imagepaths = []
for dirpath, dirnames, filenames in os.walk('Satellite Images of different areas in delhi'):
    for filename in filenames:
        # print(filename)
        imagepaths.append(os.path.join(dirpath, filename))


if __name__ == "__main__":
    # ITERATE OVER ALL IMAGES
    for i, imgpath in enumerate(imagepaths):

        # GET IMAGE AND RESIZE
        bgrimg = cv2.imread(imgpath)
        bgrimg = cv2.resize(bgrimg, (800, 600), interpolation=cv2.INTER_CUBIC)

        start = time.time()
        refined3d = mask_roofs(bgrimg)
        print("time taken by image {1}: {0:.4f} sec".format(time.time() - start, i))

        # DISPLAY RESULTS
        sbgrimg = sharpen(bgrimg)
        stacked3d = np.hstack((sbgrimg, refined3d))
        cv2.imshow("3D", stacked3d)

        # WRITE IMAGES TO DISK
        #cv2.imwrite('rooftops_mask/{}.jpg'.format(i), stacked3d)

        k = cv2.waitKey(0)
        if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()