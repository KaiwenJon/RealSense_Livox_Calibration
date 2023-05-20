import argparse
import os
import sys
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kitti_util
import numpy as np
import scipy.misc as ssc
import cv2
import matplotlib.pyplot as plt


def generate_dispariy_from_velo(pc_velo, height, width, calib):
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:, 0] < width - 1) & (pts_2d[:, 0] >= 0) & \
               (pts_2d[:, 1] < height - 1) & (pts_2d[:, 1] >= 0)
    fov_inds = fov_inds & (pc_velo[:, 0] > 2)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)
    depth_map = np.zeros((height, width)) - 1
    imgfov_pts_2d = np.round(imgfov_pts_2d).astype(int)
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        depth_map[int(imgfov_pts_2d[i, 1]), int(imgfov_pts_2d[i, 0])] = depth
    return depth_map


if __name__ == '__main__':

    # Load lidar files
    lidar = np.load('point_cloud2_to_xyz_one_minute_long_no_tf.npy')
    print(type(lidar), lidar.shape)

    # Load calibration values
    calib = kitti_util.Calibration('/fake_path')
    height = 309
    width = 309
    depth_map = generate_dispariy_from_velo(lidar, height, width, calib)
    print("Depth map shape: ", depth_map.shape)
    
    
    for i in range(len(depth_map[0])):
        for j in range(len(depth_map)):
            print(depth_map[i, j])
    
    plt.imshow(depth_map, cmap='hot', interpolation='nearest')
    plt.show()

    # print(depth_map)
    # print(np.min(depth_map))
    depth_map = depth_map + abs(np.min(depth_map))
    depth_map = depth_map/np.linalg.norm(depth_map)
    depth_map = depth_map * 255
    
    cv2.imshow('disparity map', depth_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # for fn in lidar_files:
    #     predix = fn[:-4]
    #     if predix not in file_names:
    #         continue
    #     calib_file = '{}/{}.txt'.format(calib_dir, predix)
    #     calib = kitti_util.Calibration(calib_file)
    #     # load point cloud
    #     lidar = np.fromfile(lidar_dir + '/' + fn, dtype=np.float32).reshape((-1, 4))[:, :3]
    #     image_file = '{}/{}.png'.format(image_dir, predix)
    #     image = ssc.imread(image_file)
    #     height, width = image.shape[:2]
    #     depth_map = generate_dispariy_from_velo(lidar, height, width, calib)
    #     np.save(depth_dir + '/' + predix, depth_map)
    #     print('Finish Depth Map {}'.format(predix))
