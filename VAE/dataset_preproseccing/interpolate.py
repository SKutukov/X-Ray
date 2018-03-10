import numpy as np
import cv2
from matplotlib import pyplot as plt

src_img = "../../00000321_003.png"
img = cv2.imread(src_img ,0)
print(img.shape)
# Initiate STAR detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
img2 = np.zeros((1024, 1024), dtype = "uint8")
 
ret,thresh = cv2.threshold(img,127,255,0)
cv2.drawContours(img, contours, -1, (0,255,0), 3)
# draw only keypoints location,not size and orientation
kpimg = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(im2),plt.show()