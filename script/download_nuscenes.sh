!/bin/bash
TARGET_DIR=/mnt/data1/nuscenes
cd $TARGET_DIR
echo 'downloading nuscenes'
# download train-val data (only keyframes)
wget -c https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval_meta.tgz
wget -c https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval01_keyframes.tgz
wget -c https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval02_keyframes.tgz
wget -c https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval03_keyframes.tgz
wget -c https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval04_keyframes.tgz
wget -c https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval05_keyframes.tgz
wget -c https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval06_keyframes.tgz
wget -c https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval07_keyframes.tgz
wget -c https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval07_keyframes.tgz
wget -c https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval08_keyframes.tgz
wget -c https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval09_keyframes.tgz
wget -c https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval10_keyframes.tgz
# download test data
wget -c https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-test_meta.tgz
wget -c https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-test_blobs.tgz
