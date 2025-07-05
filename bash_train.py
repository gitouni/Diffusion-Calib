import subprocess

if __name__ == "__main__":
    script_path = 'train_nlsd.py'
    python_path = '/home/bit/anaconda3/envs/pytorch/bin/python'
    dataset_config = 'cfg/dataset/nusc.yml'
    mode_config = 'cfg/mode/nlsd.yml'
    common_config = 'cfg/common.yml'
    model_configs = ['cfg/model/rggnet_nusc.yml', 'cfg/model/calibnet.yml', 'cfg/model/lccnet.yml', 'cfg/model/lccraft_large.yml']
    for model_config in model_configs:
        process = subprocess.Popen([python_path, script_path,'--dataset_config', dataset_config,'--model_config', model_config, '--common_config', common_config, '--mode_config', mode_config])
        process.wait()  # must be serialized
    # script_path = 'train_naive.py'
    # mode_config = 'cfg/mode/naiter.yml'
    # for model_config in model_configs:
    #     process = subprocess.Popen([python_path, script_path,'--dataset_config', dataset_config,'--model_config', model_config, '--common_config', common_config, '--mode_config', mode_config])
    #     process.wait()  # must be serialized
    script_path = 'train.py'
    mode_config = 'cfg/mode/lsd.yml'
    for model_config in model_configs:
        process = subprocess.Popen([python_path, script_path,'--dataset_config', dataset_config,'--model_config', model_config, '--common_config', common_config, '--mode_config', mode_config])
        process.wait()  # must be serialized