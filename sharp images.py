"""
Title: Google Maps Image Sharpening
Author: Jitender Singh Virk (Virksaab)
Date created: 24 July, 2018
Last Modified: 26 July, 2018
"""

import numpy as np
import cv2
import os
import time

# GET ALL IMAGES PATH FROM FOLDERS
imagepaths = []
for dirpath, dirnames, filenames in os.walk('Satellite Images of different areas in delhi'):
    for filename in filenames:
        # print(filename)
        imagepaths.append(os.path.join(dirpath, filename))
        
# ITERATE OVER ALL IMAGES
for i, imgpath in enumerate(imagepaths):
    # GET IMAGE AND RESIZE
    bgrimg = cv2.imread(imgpath)
    bgrimg = cv2.resize(bgrimg, (800, 600), interpolation=cv2.INTER_CUBIC)

    start = time.time()

    # SHARPEN
    kernel_sharp = np.array(([-2, -2, -2], [-2, 17, -2], [-2, -2, -2]), dtype='int')
    sbgrimg = cv2.filter2D(bgrimg, -1, kernel_sharp)
    stacked3d = np.hstack((bgrimg, sbgrimg))
    print("time taken: {:.4f} sec by image {}".format(time.time() - start, i))
    # cv2.imshow("SHARPEN", stacked3d)
    cv2.imwrite('sharpened_images/{}.jpg'.format(i), stacked3d)


    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()