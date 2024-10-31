import subprocess

if __name__ == "__main__":
    dataset_config = 'cfg/dataset/kitti_large.yml'
    multirange_config = 'cfg/dataset/multirange.yml'
    model_configs = ['cfg/multirange_model/calibnet.yml', 'cfg/multirange_model/lccnet.yml', 'cfg/multirange_model/lccraft_large.yml', 'cfg/multirange_model/lccraft_small.yml', 'cfg/multirange_model/main.yml', 'cfg/multirange_model/main_donly.yml', 'cfg/multirange_model/main_ponly.yml', 'cfg/multirange_model/rggnet.yml']
    stages = [0, 1, 2]
    for model_config in model_configs:
        for stage in stages:
            process = subprocess.Popen(['python','test_mr.py','--dataset_config',dataset_config,'--model_config',model_config,'--multirange_config',multirange_config])
            process.wait()  # must be serialized