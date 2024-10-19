import subprocess

if __name__ == "__main__":
    dataset_config = 'cfg/dataset/kitti_regular.yml'
    model_configs = ['cfg/model/main_donly.yml','cfg/model/main_ponly.yml','cfg/model/main.yml']
    sd_model_configs = ['cfg/sd_model/main_donly_sd.yml','cfg/model/rggnet_sd.yml','cfg/model/lccnet_sd.yml','cfg/model/lccraft_sd.yml']
    for model_config in model_configs:
        process = subprocess.Popen(['python','train.py','--dataset_config',dataset_config,'--model_config',model_config])
        process.wait()  # must be serialized
    for model_config in model_configs:
        process = subprocess.Popen(['python','test.py','--dataset_config',dataset_config,'--model_config',model_config,'--model_type','iterative','--iters','1'])
        process.wait()  # must be serialized
    for model_config in model_configs:
        process = subprocess.Popen(['python','test.py','--dataset_config',dataset_config,'--model_config',model_config,'--model_type','iterative','--iters','10'])
        process.wait()  # must be serialized
    for model_config in model_configs:
        process = subprocess.Popen(['python','test.py','--dataset_config',dataset_config,'--model_config',model_config,'--model_type','diffusion'])
        process.wait()  # must be serialized
    for model_config in sd_model_configs:
        process = subprocess.Popen(['python','test_se3diff.py','--dataset_config',dataset_config,'--model_config',model_config])
        process.wait()  # must be serialized