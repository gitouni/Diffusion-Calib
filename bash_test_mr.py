import subprocess
import os
if __name__ == "__main__":
    dataset_config = 'cfg/dataset/kitti_large.yml'
    multirange_config = 'cfg/dataset/mr_5.yml'
    model_configs = ['cfg/multirange_model/lccraft_small.yml','cfg/multirange_model/lccraft_large.yml']
    stages = [0, 1, 2, 3, 4]
    shutdown_cmd = 'shutdown -h now'
    for model_config in model_configs:
        for stage in stages:
            process = subprocess.Popen(['python','test_mr.py','--dataset_config',dataset_config,'--model_config',model_config,'--multirange_config',multirange_config])
            process.wait()  # must be serialized
    # os.system(shutdown_cmd) # shutdown !!!!!!!!!!!!!