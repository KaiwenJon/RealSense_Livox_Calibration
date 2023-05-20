""" 

Auth: Nathan Hung 01/20/2023

This script is used to analyze how much the depth image should resize and shift so that it can align with the realsense img while maintaining the correct size for FADNET to train.

TODO:
- resize the depth map
- align the depth map to the original size of the left img
- convert both the depth map and the left img to 768x384

"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as R
from scipy import ndimage
from matplotlib.patches import Circle, Rectangle


def align_depth_to_stereo(itr, left_img, right_img, depth_img):
    ## VARIABLES TO SHIFT THE DEPTH IMAGE TO ALIGN WITH THE REALSENSE IMAGE
    RESIZE_DEPTH_MAP = [1302, 1042] # the dimension for depth map to resize down
    REALSENSE_CENTER = [640, 360] # center location of realsense img
    DEPTH_MAP_CENTER = [631, 396] # center location of depth_map
    SHIFT_X = DEPTH_MAP_CENTER[0] - int(RESIZE_DEPTH_MAP[0]/2) #aligning center x and y
    SHIFT_Y = DEPTH_MAP_CENTER[1] - int(RESIZE_DEPTH_MAP[1]/2)
    DESIRED_W = 768
    DESIRED_H = 384


    # depth_img = cv2.resize(depth_img, (665, 532))
    depth_img = cv2.resize(depth_img, (RESIZE_DEPTH_MAP[0], RESIZE_DEPTH_MAP[1]))
    # plt.imshow(depth_img)
    # plt.show()
    depth_img = depth_img #*10 #*10  for saving the img
    # cv2.imshow('depth_img', (depth_img/np.linalg.norm(depth_img) * 255)) # normalize and * 255 for displaying
    # cv2.imwrite('Computed Depth Map.png', depth_img)
    # cv2.waitKey(0)
    print("Depth Map dim: ", depth_img.shape)

    # crop the realsense img to 768 and 384 from the center of the img
    # left_img = np.load('results/left_img_trial_1.npy')
    # cv2.imshow('left img', left_img)
    # cv2.imwrite('left img.png', left_img)
    print("IntelRealsense dim: ", left_img.shape)

    ## SHIFT THE DEPTH IMAGE CENTER TO ALIGN WITH THE PADDED REALSENSE IMAGE
    M = np.float32([[1, 0, SHIFT_X],
                    [0, 1, SHIFT_Y]])

    shifted_depth_img = cv2.warpAffine(depth_img, M, (depth_img.shape[1], depth_img.shape[0]))
    print("shifted depth img size: ", shifted_depth_img.shape)
    # fig, ax = plt.subplots(1)
    # ax.imshow(shifted_depth_img)
    # plt.imshow(shifted_depth_img)
    circ = Circle((REALSENSE_CENTER[0], REALSENSE_CENTER[1]), 3, color='red')
    # ax.add_patch(circ)
    start_pt = (int(REALSENSE_CENTER[0] - DESIRED_W/2), int(REALSENSE_CENTER[1] - DESIRED_H/2))
    end_pt = (int(REALSENSE_CENTER[0] + DESIRED_W/2), int(REALSENSE_CENTER[1] + DESIRED_H/2))
    rect = Rectangle(start_pt, 768, 384, linewidth=1, edgecolor='r', facecolor='none')
    # ax.add_patch(rect)
    # plt.show()
    # cv2.imshow('shifted depth map', shifted_depth_img)
    ## Crop the data into 768 by 384 from the center
    cropped_shifted_depth_img = shifted_depth_img[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0]].copy() # row x col
    print("cropped points", start_pt[1], end_pt[1], start_pt[0], end_pt[0])
    # plt.figure()
    # plt.imshow(cropped_shifted_depth_img)
    # plt.show()
    plt.imsave("results/Cropped_shifted_depth_img.png",cropped_shifted_depth_img)



    ## ADD PADDING TO THE REALSESNSE IMAGE TO ENSURE THE SAME SIZE
    padded_left_img = cv2.copyMakeBorder(left_img, 0, shifted_depth_img.shape[0] - left_img.shape[0], 0, shifted_depth_img.shape[1] - left_img.shape[1], cv2.BORDER_CONSTANT, value=[255, 255, 0])
    padded_right_img = cv2.copyMakeBorder(right_img, 0, shifted_depth_img.shape[0] - right_img.shape[0], 0, shifted_depth_img.shape[1] - right_img.shape[1], cv2.BORDER_CONSTANT, value=[255, 255, 0])
    
    #padded_left_img = padded_left_img * 10 #255plt.figure()
    print("padded left img size: ", padded_left_img.shape)
    # cv2.imshow('padded left img', padded_left_img[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0]])
    processed_padded_left_img = padded_left_img[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0]]
    processed_padded_right_img = padded_right_img[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0]]    
    cv2.imwrite('results/Cropped_padded_left_img'+str(itr)+'.png', padded_left_img[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0]])
    cv2.imwrite('results/Cropped_padded_right_img'+str(itr)+'.png', padded_right_img[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0]])
    # cv2.waitKey(0)


    ## OVERLAY TWO IMAGES
    print(type(left_img), type(depth_img))
    padded_left_img = padded_left_img.astype('float64')
    dst = cv2.addWeighted(padded_left_img, 0.2, shifted_depth_img, 1.0, 125)
    dst = dst / np.linalg.norm(dst)
    dst = dst * 255
    # cv2.imshow('Blended img', dst)

    # Draw a rectangle to visualize how to cut?
    color = (0, 255, 0) # green
    thickness = 2 # px
    start_pt = (int(REALSENSE_CENTER[0] - DESIRED_W/2), int(REALSENSE_CENTER[1] - DESIRED_H/2))
    end_pt = (int(REALSENSE_CENTER[0] + DESIRED_W/2), int(REALSENSE_CENTER[1] + DESIRED_H/2))
    print(start_pt, end_pt)
    rectangle_img = cv2.rectangle(dst, start_pt, end_pt, color, thickness)
    cv2.imshow('draw rectangle', rectangle_img)
    image = cv2.circle(rectangle_img, (REALSENSE_CENTER[0],REALSENSE_CENTER[1]), radius=5, color=(0, 0, 255), thickness=-1)
    cv2.imshow('draw center dot', image)




    ## DISPLAY AND SAVE IMAGE
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite('Saved_blended.png', dst)
    # cv2.imwrite('Saved_left.png', padded_left_img)
    # cv2.imwrite('Saved_shifted.png', shifted_depth_img)
    
    return cropped_shifted_depth_img, processed_padded_left_img


if __name__ == '__main__':
    pass