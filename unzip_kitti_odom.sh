#!/bin/bash
TARGET_DIR=/mnt/data1/kitti/odometry/
cd $TARGET_DIR
unzip data_odometry_color.zip
unzip -o data_odometry_velodyne.zip
unzip -o data_odometry_gray.zip
unzip -o data_odometry_poses.zip
unzip -o data_odometry_calib.zip