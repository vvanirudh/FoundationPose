## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

import pyrealsense2 as rs
import numpy as np
import cv2
import json
import time
import os

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []

    for c in corners:
        nada, R, t = cv2.solvePnP(
            marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE
        )
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return np.array(rvecs).squeeze(), np.array(tvecs).squeeze(), trash

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == "RGB Camera":
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == "L500":
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Get the absolute path to the subfolder
script_dir = os.path.dirname(os.path.abspath(__file__))
subfolder_depth = os.path.join(script_dir, "out/depth")
subfolder_rgb = os.path.join(script_dir, "out/rgb")
subfolder_depth_unaligned = os.path.join(script_dir, "out/depth_unaligned")
subfolder_rgb_unaligned = os.path.join(script_dir, "out/rgb_unaligned")
subfolder_artag_pose = os.path.join(script_dir, "out/ar_tag_pose")
subfolder = os.path.join(script_dir, "out")

# Check if the subfolder exists, and create it if it does not
if not os.path.exists(subfolder_depth):
    os.makedirs(subfolder_depth)
if not os.path.exists(subfolder_rgb):
    os.makedirs(subfolder_rgb)
if not os.path.exists(subfolder_depth_unaligned):
    os.makedirs(subfolder_depth_unaligned)
if not os.path.exists(subfolder_rgb_unaligned):
    os.makedirs(subfolder_rgb_unaligned)
if not os.path.exists(subfolder_artag_pose):
    os.makedirs(subfolder_artag_pose)

# Create all

RecordStream = False

# AR tag
aruco_dict_type = cv2.aruco.DICT_6X6_250
dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
marker_size = 0.024  # 2.4cm
axes_size = 0.05

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = (
            aligned_frames.get_depth_frame()
        )  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        unaligned_depth_frame = frames.get_depth_frame()
        unaligned_color_frame = frames.get_color_frame()

        # Get instrinsics from aligned_depth_frame
        intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        intrinsic_matrix = np.array([
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ])
        distortion_coefficients = np.asanyarray(intrinsics.coeffs)

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack(
            (depth_image, depth_image, depth_image)
        )  # depth image is 1 channel, color is 3 channels
        bg_removed = np.where(
            (depth_image_3d > clipping_distance) | (depth_image_3d <= 0),
            grey_color,
            color_image,
        )

        unaligned_depth_image = np.asanyarray(unaligned_depth_frame.get_data())
        unaligned_rgb_image = np.asanyarray(unaligned_color_frame.get_data())

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        ar_image = np.asanyarray(color_frame.get_data()).copy()
        marker_poses = []
        if len(corners) > 0:
            for i in range(len(ids)):
                rvec, tvec, _ = my_estimatePoseSingleMarkers(
                    corners[i], marker_size, intrinsic_matrix, distortion_coefficients
                )
                # Draw a square around the markers
                cv2.aruco.drawDetectedMarkers(ar_image, corners)

                # Draw Axis
                cv2.drawFrameAxes(
                    ar_image, intrinsic_matrix, distortion_coefficients, rvec, tvec, axes_size
                )

                marker_pose = np.zeros((4, 4))
                marker_pose[:3, :3], _ = cv2.Rodrigues(rvec)
                marker_pose[:3, -1] = tvec
                marker_pose[3, 3] = 1.0
                marker_poses.append(marker_pose)


        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )
        images = np.hstack((color_image, depth_colormap, ar_image))

        cv2.namedWindow("Align Example", cv2.WINDOW_NORMAL)
        cv2.imshow("Align Example", images)

        key = cv2.waitKey(1)

        # Start saving the frames if space is pressed once until it is pressed again
        if key & 0xFF == ord(" "):
            if not RecordStream:
                time.sleep(0.2)
                RecordStream = True

                with open(os.path.join(script_dir, "out/cam_K.txt"), "w") as f:
                    f.write(f"{intrinsics.fx} {0.0} {intrinsics.ppx}\n")
                    f.write(f"{0.0} {intrinsics.fy} {intrinsics.ppy}\n")
                    f.write(f"{0.0} {0.0} {1.0}\n")

                np.save(os.path.join(subfolder, "calibration_matrix.npy"), intrinsic_matrix)
                np.save(os.path.join(subfolder, "distortion_coefficients.npy"), intrinsics.coeffs)

                print("Recording started")
            else:
                RecordStream = False
                print("Recording stopped")

        if RecordStream:
            framename = int(round(time.time() * 1000))

            # Define the path to the image file within the subfolder
            image_path_depth = os.path.join(subfolder_depth, f"{framename}.png")
            image_path_rgb = os.path.join(subfolder_rgb, f"{framename}.png")
            image_path_depth_unaligned = os.path.join(subfolder_depth_unaligned, f"{framename}.png")
            image_path_rgb_unaligned = os.path.join(subfolder_rgb_unaligned, f"{framename}.png")
            ar_tag_pose_path = os.path.join(subfolder_artag_pose, f"{framename}_ar_tag_pose.npy")


            cv2.imwrite(image_path_depth, depth_image)
            cv2.imwrite(image_path_rgb, color_image)
            cv2.imwrite(image_path_depth_unaligned, unaligned_depth_image)
            cv2.imwrite(image_path_rgb_unaligned, unaligned_rgb_image)
            # [num_markers, 4, 4]
            np.save(ar_tag_pose_path, np.array(marker_poses))

        # Press esc or 'q' to close the image window
        if key & 0xFF == ord("q") or key == 27:

            cv2.destroyAllWindows()

            break
finally:
    pipeline.stop()