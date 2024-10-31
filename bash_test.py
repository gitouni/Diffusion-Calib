import subprocess

if __name__ == "__main__":
    dataset_config = 'cfg/dataset/kitti_large.yml'
    model_configs = ['cfg/unipc_model/calibnet.yml', 'cfg/unipc_model/lccnet.yml', 'cfg/unipc_model/lccraft_large.yml', 'cfg/unipc_model/lccraft_small.yml', 'cfg/unipc_model/main.yml', 'cfg/unipc_model/main_donly.yml', 'cfg/unipc_model/main_ponly.yml', 'cfg/unipc_model/rggnet.yml']
    sd_model_configs = ['cfg/unipc_sd_model/calibnet_sd.yml', 'cfg/unipc_sd_model/lccnet_sd.yml', 'cfg/unipc_sd_model/lccraft_large_sd.yml', 'cfg/unipc_sd_model/lccraft_sd.yml', 'cfg/unipc_sd_model/lccraft_small_sd.yml', 'cfg/unipc_sd_model/main_donly_sd.yml', 'cfg/unipc_sd_model/main_ponly_sd.yml', 'cfg/unipc_sd_model/main_sd.yml', 'cfg/unipc_sd_model/rggnet_sd.yml']
    # for model_config in model_configs:
    #     process = subprocess.Popen(['python','test.py','--dataset_config',dataset_config,'--model_config',model_config,'--model_type','iterative','--iters','1'])
    #     process.wait()  # must be serialized
    for model_config in model_configs:
        process = subprocess.Popen(['python','test.py','--dataset_config',dataset_config,'--model_config',model_config,'--model_type','iterative','--iters','10'])
        process.wait()  # must be serialized
    # for model_config in model_configs:
    #     process = subprocess.Popen(['python','test.py','--dataset_config',dataset_config,'--model_config',model_config,'--model_type','diffusion'])
    #     process.wait()  # must be serialized
    # for model_config in sd_model_configs:
    #     process = subprocess.Popen(['python','test_se3diff.py','--dataset_config',dataset_config,'--model_config',model_config])
    #     process.wait()  # must be serialized