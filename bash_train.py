import subprocess

if __name__ == "__main__":
    script_paths = ['train.py','train_surrogate.py','train_naive.py']
    python_path = '/home/bit/anaconda3/envs/pytorch/bin/python'
    dataset_config = 'cfg/dataset/nusc.yml'
    model_configs = ['cfg/lsd_model/main.yml','cfg/sd_model/main_sd.yml']
    # model_configs = ['cfg/sd_model/rggnet_sd.yml','cfg/sd_model/calibnet_sd.yml','cfg/sd_model/lccnet_sd.yml','cfg/sd_model/lccraft_large_sd.yml','cfg/sd_model/lccraft_small_sd.yml']
    for script_path, model_config in zip(script_paths, model_configs):
        process = subprocess.Popen([python_path, script_path,'--dataset_config',dataset_config,'--model_config',model_config])
        process.wait()  # must be serialized