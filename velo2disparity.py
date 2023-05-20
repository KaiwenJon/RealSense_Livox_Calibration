"""
Author: Kevin Chuang 01/19/2023 
Modified: Nathan Hung 01/24/2023

"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as R
from scipy import ndimage
from align_depth_and_realsense import *



def convert_velo2depth(itr, lidar):
    # Tune parameters tz (camera offset from the center of lidar) and f (focal length)
    # tz = 5, f = 1200
    # tz = 1, f = 700


    """ 
    tx is horizontal
    tz is into the page - higher tz means camera is further out of the page
    """

    tx = 0.0654 #0.3654 #0.0654 # 0 
    ty = 0.0153 # 0
    tz = 0.09 # 0.00001
    fx = 1000 #1200 # higher f, zooms in effect, background is closer
    fy = fx # lower fy, stretches horizontally
    ox = 1000
    oy = 800
    
    # tx = 0.0654 #0.3654 #0.0654 # 0 
    # ty = 0.0153 # 0
    # tz = 0.09 # 0.00001
    # fx = 651.3909 #1200 # higher f, zooms in effect, background is closer
    # fy = fx # lower fy, stretches horizontally
    # ox = 643 #1000
    # oy = 401


    # lidar = np.load('point_cloud2_to_xyz_one_minute_long_no_tf.npy')
    # lidar = np.load('point_cloud2_to_xyz_for_RealSense.npy')
    # lidar = np.load('point_cloud2_to_xyz_livox_realsense_v3.npy')
    # lidar = np.load('results/lidar_ptcld2_trial_1.npy')
    lidar = lidar.T

    # Build up extrinsic parameters (relative transformation between lidar and camera)
    r = R.from_euler('y', -90, degrees=True)
    r2 = R.from_euler('z', 90, degrees=True)
    t = np.array([[tx, ty, tz]])
    extrinsic = np.hstack((r2.as_matrix() @ r.as_matrix(), t.T))
    extrinsic = np.vstack((extrinsic, np.array([0, 0, 0, 1])))

    # Build up intrinsic parameters
    intrinsic = np.array([[fx ,   0., ox, 0],
                        [ 0.,   fy , oy, 0],
                        [ 0.,   0., 1., 0]])

    lidar = np.vstack((lidar, np.ones((1, lidar.shape[1]))))
    lidar_proj = intrinsic @ extrinsic @ lidar
    d = lidar_proj[2, :]
    lidar_proj = lidar_proj / d
    image = np.ones((oy*2, ox*2))*np.max(d)
    left_img = np.ones((oy*2, ox*2))*np.max(d)
    right_img = np.ones((oy*2, ox*2))*np.max(d)

    u = lidar_proj[0, :]
    v = lidar_proj[1, :]
    for point in zip(u, v, d):
        u_pix = round(point[1])
        v_pix = round(point[0])
        if(u_pix < 0 or u_pix >= image.shape[0] or v_pix < 0 or v_pix >= image.shape[1]):
            continue
        
        depth_z = min(image[u_pix][v_pix], point[2])
        image[u_pix][v_pix] = depth_z

    image = ndimage.median_filter(image, size=5)
    fig = plt.figure()
    # plt.imshow(image)
    # plt.title("Depth map from lidar points")
    # plt.colorbar()
    np.save('data/lidar_depth_trial_'+ str(itr), image)

    # plt.show()

    img_left = np.load('data/left_img_trial_'+str(itr)+'.npy')

    img_right = np.load('data/right_img_trial_'+str(itr)+'.npy')

    # cv2.imshow('RealSense', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite('left_RealSense.png',img)
    
    return image, img_left, img_right # depth and img

def write_ply(fn, verts, colors):
    ply_header = '''ply
    format ascii 1.0t
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    out_colors = colors.copy()
    verts = verts.reshape(-1, 3)
    verts = np.hstack([verts, out_colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
        
if __name__ == '__main__':
    tx = 0.0654 #0.3654 #0.0654 # 0 
    ty = 0.0153 # 0
    tz = 0.09 # 0.00001
    fx = 1000 #1200 # higher f, zooms in effect, background is closer
    fy = fx # lower fy, stretches horizontally
    ox = 1000
    oy = 800
    w = 768 #1280
    h = 384 #720
    f = fx
    b = 59.999999865889549 # mm
    FOV = 91.2 # in degrees
    fx_pix = (w*0.5)/np.tan(FOV * 0.5 * np.pi/180)
    
    fx_pix_depth = (1302*0.5)/np.tan(FOV * 0.5 * np.pi/180)
    
    print('Converting lidar trials to depth...')

    
    trial_itr = 25

    while(trial_itr < 33):
        lidar = np.load('data/lidar_ptcld2_trial_' + str(trial_itr) + '.npy')
        lidar_img, left_img, right_img = convert_velo2depth(trial_itr,lidar)
    
        processed_depth_img, processed_left_img = align_depth_to_stereo(trial_itr, left_img, right_img, lidar_img)
        disparity_img = (b*fx_pix_depth) / processed_depth_img
        print('disp min max', np.min(disparity_img), np.max(disparity_img))
        disparity_img = ((disparity_img/np.linalg.norm(disparity_img)) * 128).astype('float32')
        cv2.imshow("disp img", disparity_img)
        plt.imsave("results/disp_img"+str(trial_itr)+".png", disparity_img)
        trial_itr += 1
    # cv2.waitKey(0)
    # plt.imshow(disparity_img)
    # plt.colorbar()
    # plt.show()
    
    
    
    Q = np.array([[1, 0, 0, -w/2],
                [0, 1, 0, -h/2],
                [0, 0, 0, fx_pix], 
                [0, 0, -1/b, 0]])
    
    # print(-w/2, -h/2, fx_pix)
    
    # print("disparity_img", disparity_img)
    # print(np.min(disparity_img), np.max(disparity_img))
    # ptcl2 = cv2.reprojectImageTo3D(disparity_img.astype('float32'), Q.astype('float32'))
    # ptcl2 = cv2.normalize(ptcl2, 0, 255)
    # points = ptcl2
    
    # opencv loads the image in BGR, so lets make it RGB
    # colors = cv2.cvtColor(stereo_img, cv2.COLOR_BGR2RGB)
    # resize to match point cloud shape
    # colors = colors.resize(-1, 3)
    # write_ply('out.ply', out_points, out_colors)
    
    #print(ptcl2)
    # points = np.zeros((disparity_img.shape[0]+disparity_img.shape[1], 3))
    # points_x = []
    # points_y = []
    # points_z = []
    # for ul in range(disparity_img.shape[0]):
    #     for vl in range(disparity_img.shape[1]):
    #         x = b*(ul - ox)/disparity_img[ul, vl]
    #         y = (b*fx*(vl - oy))/(fy*(disparity_img[ul, vl]))
    #         z = (b*fx)/(disparity_img[ul, vl])
    
    #         points_x.append(x)
    #         points_y.append(y)
    #         points_z.append(z)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(ptcl2[0], ptcl2[1], ptcl2[2])
    # plt.show()
    
    #cv2.imshow("reprojected", ptcl2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    # TODO
    # Add search through all the trials and convert thme in to depth images
    
    
    #reflect on x axis
    #reflect_matrix = np.identity(3)
    #reflect_matrix[0] *= -1
    #points = np.matmul(points,reflect_matrix)

    #extract colors from image
    #colors = cv2.cvtColor(processed_left_img, cv2.COLOR_BGR2RGB)

    #filter by min disparity
    #mask = disparity_img > disparity_img.min()
    #out_points = points[mask]
    #out_colors = colors[mask]

    #filter by dimension
    #idx = np.fabs(out_points[:,0]) < 4.5
    #out_points = out_points[idx]
    #out_colors = out_colors.reshape(-1, 3)
    #out_colors = out_colors[idx]

    #Save as polygon/mesh files and read it using MeshLab
    #write_ply('results/out_4_with_fx_pix.ply', out_points, out_colors)
    #print('%s saved' % 'out.ply')
