# RealSense_Livox
- Calibration between RealSense camera and LiDAR (Livox) using computer vision techniques (PnP method)
- Image and LiDAR data Collection using ROS2 subscriber
```
python3 estimate_tf_between_livox_realsense.py
```
and follow the instructions in the .py script.

### Uncalibrated - Raw image and raw LiDAR map
![image](https://github.com/KaiwenJon/RealSense_Livox/assets/70893513/0d0022b3-b30c-4e41-ae62-38427785d557)

### Calibrated - image and LiDAR map are both viewed from the origin of real-sense camera
![image](https://github.com/KaiwenJon/RealSense_Livox/assets/70893513/7516ccb1-0e7b-444a-8813-a5965c79a381)
