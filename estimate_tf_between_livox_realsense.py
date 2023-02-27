import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import cv2
from scipy import ndimage

def convert_velo2depth(extrinsic, intrinsic, lidar):
    ox = int(intrinsic[0][2])
    oy = int(intrinsic[1][2])
    lidar = lidar.T
    lidar = np.vstack((lidar, np.ones((1, lidar.shape[1]))))
    lidar_proj = intrinsic @ extrinsic @ lidar
    d = lidar_proj[2, :]
    d[d == 0] = 1e-10
    lidar_proj = lidar_proj / d
    image = np.ones((oy*2, ox*2))*np.max(d)

    u = lidar_proj[0, :]
    v = lidar_proj[1, :]
    for point in zip(u, v, d):
        if(point[2] < 0):
            # points are behind the camera, don't show it
            continue
        u_pix = round(point[1])
        v_pix = round(point[0])
        if(u_pix < 0 or u_pix >= image.shape[0] or v_pix < 0 or v_pix >= image.shape[1]):
            #points are outside the image (h,w), don't show it
            continue
        depth_z = min(image[u_pix][v_pix], point[2])
        image[u_pix][v_pix] = depth_z

    image = ndimage.median_filter(image, size=5)
    # fig = plt.figure()
    # fig.add_subplot(2, 1, 1)
    # plt.imshow(image)
    # plt.title("Depth map from lidar points")
    # plt.colorbar()
    # np.save('data/lidar_depth_trial_1', image)


    # img = np.load('data/data/RealSense_left_img.npy')
    # fig.add_subplot(2, 1, 2)
    # plt.imshow(img)
    # plt.colorbar()
    # plt.show()
    # cv2.imshow('RealSense', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite('RealSense.png',img)
    
    return image # depth and img

##### Dataset: left image from realsense & pointcloud from livox #####
left_img = np.load('data/left_img_trial_1.npy')
lidar = np.load('data/lidar_ptcld2_trial_1.npy')
print("left_img shape:", left_img.shape)
h, w = left_img.shape
##### Step1: Using pointcloud, construct a depth map, where the perspective is from the origin of livox. Therefore, set tx,ty,tx to 0.
# Build up extrinsic parameters (relative transformation between lidar and camera)
r2 = R.from_euler('z', 90, degrees=True)
r = R.from_euler('y', -90, degrees=True)
# t = np.array([[0, 0, tz]])
t = np.array([[0, 0, 0]])
extrinsic = np.hstack((r2.as_matrix() @ r.as_matrix(), t.T))
extrinsic = np.vstack((extrinsic, np.array([0, 0, 0, 1])))

# Build up intrinsic parameters (should follow exactly the spec of realsense)
ox = w/2 # 640/2
oy = h/2 # 360/2
FOV = 91.2
fx = (w*0.5)/np.tan(FOV * 0.5 * np.pi/180)
fy = fx
intrinsic = np.array([[fx ,   0., ox, 0.],
                    [ 0.,   fy , oy, 0.],
                    [ 0.,   0., 1., 0.]])
fig = plt.figure()
fig.add_subplot(2, 1, 1)
lidar_img = convert_velo2depth(lidar=lidar, extrinsic=extrinsic, intrinsic=intrinsic)
# np.save("lidar_img.npy", lidar_img)
# lidar_img = np.load("lidar_img.npy")
plt.title("Uncalibrated lidar depth map and realsense image")
plt.imshow(lidar_img)
plt.colorbar()

fig.add_subplot(2, 1, 2)
plt.imshow(left_img)
plt.colorbar()

####### Step2, Using the depth map and left image, estimate the extrinsic matrix ###############
# Goal: We want to estimate the tf from lidar to realsense, using PnP method.
# Assume the origin of the world coordinate is at lidar's center. 
# We need two things to get Pnp work.
# 1. a set of 2D points on the image of realsense.
# 2. a set of 3D points in the world coordinate corresponding to those 2D points.
# 3. PnP will give us the transformation between world coordinate and realsense, which is exactly what we want.

# Let's pick some obvious feature points, and see their position in realsense(2D) and lidar(3D)
print("Manually pick some features and see where they are located in two pictures.")
plt.show()
# Manually type in the below two vectors.

lidar_img_points = np.array([[150, 228], 
                            [258, 208],
                            [429, 142],
                            [509, 229],
                            [474, 242],
                            [146, 80]],dtype=np.float64)
realsense_img_points = np.array([[150, 250], 
                                [260, 226],
                                [433, 162],
                                [518, 252],
                                [479, 264],
                                [151, 102]],dtype=np.float64)
# Done! Let's run PnP!


object_points = np.zeros((0, 3))
for [px, py] in lidar_img_points:
    z = lidar_img[int(py)][int(px)]
    x = z*(px-ox)/fx
    y = z*(py-oy)/fy
    object_points = np.vstack([object_points, np.array([z, -x, -y])])
    
print("The features you picked in world coordinate(lidar coord), which is Z pointing to the ceiling, and X forward\n", object_points)
print("The features you picked in realsense image coord, X rightward, Y downward.\n", realsense_img_points)


_, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(object_points, realsense_img_points, intrinsic[:, :3], 0)
rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
extrinsic_matrix = np.concatenate((rotation_matrix, translation_vector), axis = 1)
extrinsic_matrix = np.vstack((extrinsic_matrix, np.array([0, 0, 0, 1])))
print("estimated extrinsic matrix\n", extrinsic_matrix)


fig = plt.figure()
fig.add_subplot(2, 1, 1)
lidar_img = convert_velo2depth(lidar=lidar, extrinsic=extrinsic_matrix, intrinsic=intrinsic)
# np.save("lidar_img.npy", lidar_img)
plt.title("Calibrated lidar depth map and realsense image")
plt.imshow(lidar_img)
plt.colorbar()

fig.add_subplot(2, 1, 2)
plt.imshow(left_img)
plt.colorbar()

plt.show()