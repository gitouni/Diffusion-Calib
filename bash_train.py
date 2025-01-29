import subprocess

if __name__ == "__main__":
    script_path = 'train_mr.py'
    python_path = '/home/ouni/miniconda3/envs/relposepp/bin/python'
    dataset_config = 'cfg/dataset/kitti_large.yml'
    mode_config = 'cfg/mode/mr_3.yml'
    common_config = 'cfg/common.yml'
    model_configs = ['cfg/model/calibnet.yml', 'cfg/model/lccnet.yml', 'cfg/model/rggnet.yml', 'cfg/model/lccraft_small.yml','cfg/model/lccraft_large.yml']
    # model_configs = ['cfg/model/lccraft_small.yml','cfg/model/lccraft_large.yml']
    for model_config in model_configs:
        process = subprocess.Popen([python_path, script_path,'--dataset_config', dataset_config,'--model_config', model_config, '--common_config', common_config, '--mode_config', mode_config])
        process.wait()  # must be serialized