"""
Nathan Hung @ 01/24/2023
This code subscribes to the two topics: Lidar and StereoCamera, converts them from ROS2 message into Numpy format.
Data is then extracted for cropp and visualization
Credit:
https://github.com/Box-Robotics/ros2_numpy
https://github.com/eric-wieser/ros_numpy
"""
from pynput import keyboard

# ros2_numpy libraries
import ros2_numpy.registry as rg
import ros2_numpy.point_cloud2 as pcl2
import ros2_numpy.image as ros_im
# import registry as rg
# import point_cloud2 as pcl2
# import image as ros_im


# ros2 libraries
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from rclpy.qos import ReliabilityPolicy, QoSProfile

# other libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import figure
import cv2

import os

cur_path = cwd = os.getcwd()
print(cur_path)


class SimpleSubscriber(Node):

    def __init__(self):
        
        # img settings
        self.HEIGHT = 1
        self.WIDTH = 9984
        
        # point cloud settings
        self.ARRAY_POSITION = 0
        self.ROW_STEP = 179712
        self.POINT_step = 18
        self.pt_cloud_xyz = np.zeros((self.WIDTH, 3))
        
        # other variables
        self.IMG_FLAG = 0
        self.TICKS_TO_SAVE = 420 #420 # change thsi based on your need, number of total ticks/frames to save for lidar
        self.count = 0 # variable to count how many ticks of the frames needed to store for pointcloud2 data
        
        self.LIDAR_NUMPY_FILENAME = 'data/lidar_ptcld2_trial_'
        self.LEFT_IMG_NUMPY_FILENAME = 'data/left_img_trial_'
        self.RIGHT_IMG_NUMPY_FILENAME = 'data/right_img_trial_'
        
        self.spaceTrigger = False
        self.img_sequence = 1
        
        super().__init__('livox_subscriber')
        ## Subscriber for Lidar
        self.subscriber = self.create_subscription(
            PointCloud2,
            '/livox/lidar',
            self.lidar_callback, 1) #QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))  # is the most used to read LaserScan data and some sensor data.
        
        # Subscriber for Stereo camera
        self.img_subscriber = self.create_subscription(
            Image,
            '/camera/infra1/image_rect_raw',
            self.left_callback, 1) #QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        
        self.img_subscriber = self.create_subscription(
            Image,
            '/camera/infra2/image_rect_raw',
            self.right_callback, 1) #QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))

        # Detect keypress
        def on_press(key):
            try:
                # print('alphanumeric key {0} pressed'.format(key.char))
                pass
            except AttributeError:
                # print('special key {0} pressed'.format(key))
                pass

        def on_release(key):
            # print('{0} released'.format(
                # key))
            if (key == keyboard.Key.space) & (self.spaceTrigger == False):
                self.spaceTrigger = True
                print('SPACE pressed, now recording...')
            else:
                print('Recording... please wait')
            
            if key == keyboard.Key.esc:
                # Stop listener
                return False

        # ...or, in a non-blocking fashion:
        listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release)
        listener.start()
        
        
        
    def left_callback(self, msg):     
        """Auto Version"""
        # rclpy.spin_once(self)
        print("img:" , self.spaceTrigger, self.IMG_FLAG)
        if self.spaceTrigger and self.IMG_FLAG < 2:
            self.IMG_FLAG += 1
            img_np = ros_im.image_to_numpy(msg)
            # save_path = os.path.join(cur_path, self.IMG_NUMPY_FILENAME+str(self.img_sequence))
            # print("save path: ", save_path)
            np.save(self.LEFT_IMG_NUMPY_FILENAME+str(self.img_sequence), img_np)
            print('left image saved...')

    def right_callback(self, msg):     
        """Auto Version"""
        # rclpy.spin_once(self)
        print("img:" , self.spaceTrigger, self.IMG_FLAG)
        if self.spaceTrigger and self.IMG_FLAG < 2:
            self.IMG_FLAG += 1
            img_np = ros_im.image_to_numpy(msg)
            # save_path = os.path.join(cur_path, self.IMG_NUMPY_FILENAME+str(self.img_sequence))
            # print("save path: ", save_path)
            np.save(self.RIGHT_IMG_NUMPY_FILENAME+str(self.img_sequence), img_np)
            print('right image saved...')

    def lidar_callback(self, msg):
        print("lidar:", self.spaceTrigger, self.IMG_FLAG)
        """Manual version"""        
        # self.pt_cloud_xyz = np.concatenate((self.pt_cloud_xyz, pcl2.pointcloud2_to_xyz_array(msg)), axis=0)
        
        # # number of counts to save the lidar data
        # if self.count == self.TICKS_TO_SAVE:
        #     np.save(self.LIDAR_NUMPY_FILENAME, self.pt_cloud_xyz)
        #     b = np.load(self.LIDAR_NUMPY_FILENAME)
        #     print(np.array_equal(self.pt_cloud_xyz, b))
        #     fig = figure()
        #     ax = fig.add_subplot(projection='3d')
        #     ax.scatter(self.pt_cloud_xyz[:,0], self.pt_cloud_xyz[:,1], self.pt_cloud_xyz[:, 2], s = 0.1, c = -self.pt_cloud_xyz[:, 0])
        #     ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 2, 1, 1])) # scale x, y, z = 1, 2, 1
        #     ax.view_init(elev= 2, azim = -180)
        #     ax.set_xlabel('x')
        #     ax.set_ylabel('y')
        #     ax.set_zlabel('z')
        #     plt.show()
        
        # self.count+=1
        
        """Auto version"""
        # if space bar is pressed, start collecting data
        if self.spaceTrigger:
            self.pt_cloud_xyz = np.concatenate((self.pt_cloud_xyz, pcl2.pointcloud2_to_xyz_array(msg)), axis=0)
        
            # After collecting N ticks of data, we save it
            # Also reset the space trigger ready for next recording
            if self.count == self.TICKS_TO_SAVE:
                np.save(self.LIDAR_NUMPY_FILENAME+str(self.img_sequence), self.pt_cloud_xyz)
                # reset values
                self.spaceTrigger = False
                self.IMG_FLAG = 0
                self.count = 0
                self.pt_cloud_xyz = np.zeros((self.WIDTH, 3))
                print("Recording done trial, ready for new trial...")
                self.img_sequence += 1 # update trial number
                # rclpy.spin_once(self)
            
            self.count+=1
            # print('Recording ticks: ', self.count)
        
        
        
        
        ### print the log info in the terminal
        # pt_cloud2 = np.asarray(msg.data)
        # pt_cloud_xyz = np.reshape(pt_cloud2,(self.row_step//3, 3)) #.astype(np.float64)
        # print(pt_cloud_xyz[:,0], pt_cloud_xyz[:,1], pt_cloud_xyz[:, 2])
        # self.get_logger().info('Feedback: {0} '.format(pt_cloud_xyz))
        
        
        
        # print(type(msg.data))
        # print(len(msg.data))
        # print(max(msg.data))
        # print(min(msg.data))
        # self.get_logger().info('I receive: "%s"' % str(msg))
        
        # pt_cloud2 = np.asarray(msg.data)
        # print(pt_cloud2)
        # is_all_zeros = not np.any(pt_cloud2) #check if all elements are 0
        # if is_all_zeros:
        # xyz_pts = np.frombuffer(msg.data, dtype=np.float64)
        # xyz_pts = np.zeros(xyz_pts.shape + (3,))
        # print(xyz_pts.shape)
        # print(np.min(xyz_pts))
        # print(np.max(xyz_pts))
        


def main(args=None):
    # initialize the ROS communication
    rclpy.init(args=args)
    # declare the node constructor
    simple_subscriber = SimpleSubscriber()
    # pause the program execution, waits for a request to kill the node (ctrl+c)
    rclpy.spin(simple_subscriber)
    # Explicity destroy the node
    simple_subscriber.destroy_node()
    # shutdown the ROS communication
    rclpy.shutdown()


if __name__ == '__main__':
    main()