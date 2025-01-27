#!/bin/bash
TARGET_DIR=/mnt/data1/kitti/odometry
cd $TARGET_DIR
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip
wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip
