""" @Nathan Hung 01/20/2023"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as R
from scipy import ndimage

"""
TODO:
- resize the depth map
- align the depth map to the original size of the left img
- convert both the depth map and the left img to 768x384
"""


depth_img = np.load('computed_depth_map_v1.npy')
print("depth img size: ", depth_img.shape)
# print(depth_img[0][0])
depth_img = cv2.resize(depth_img, (665, 532))
# print(np.min(depth_img), np.max(depth_img))
# depth_img = depth_img/np.linalg.norm(depth_img)
depth_img = depth_img*10 #*10  for saving the img
# print(np.min(depth_img), np.max(depth_img))
# print(depth_img.shape)



# crop the realsense img to 768 and 384 from the center of the img
left_img = np.load('Livox_RealSense_left_img_v2.npy')
print("img ori_size: ", left_img.shape)

center = left_img.shape
width = 768
height = 384
x = center[1]/2 - width/2
y = center[0]/2 - height/2


#Shift the depth img:
M = np.float32([[1, 0, -13],
                [0, 1, -66]])

shifted_depth_img = cv2.warpAffine(depth_img, M, (depth_img.shape[1], depth_img.shape[0]))
print("shifted depth img size: ", shifted_depth_img.shape)
cv2.imshow('depth map', shifted_depth_img   )


padded_left_img = cv2.copyMakeBorder(left_img, 0, shifted_depth_img.shape[0] - left_img.shape[0], 0, shifted_depth_img.shape[1] - left_img.shape[1], cv2.BORDER_CONSTANT, value=[255, 255, 0])
# padded_left_img = padded_left_img / np.linalg.norm(padded_left_img)
padded_left_img = padded_left_img * 10 #255
print("padded left img size: ", padded_left_img.shape)

## overlay two imgs
print(type(left_img), type(depth_img))
padded_left_img = padded_left_img.astype('float64')
dst = cv2.addWeighted(padded_left_img, 0.2, shifted_depth_img, 1.0, 125)
dst = dst / np.linalg.norm(dst)
dst = dst * 255
cv2.imshow('Blended img', dst)


# cv2.imshow('shifted_depth_img map', shifted_depth_img)
# cv2.imshow('depth map', depth_img)
# cv2.imshow('Padded RealSense', padded_left_img)
# cv2.imshow('add', img_add)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite('Saved_blended.png', dst)
# cv2.imwrite('Saved_left.png', padded_left_img)
# cv2.imwrite('Saved_shifted.png', shifted_depth_img)
