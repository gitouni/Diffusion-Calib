import subprocess

if __name__ == "__main__":
    dataset_config = 'cfg/dataset/kitti_large.yml'
    multirange_config = 'cfg/dataset/multirange.yml'
    model_configs = ['cfg/multirange_model/lccraft_small.yml','cfg/multirange_model/lccraft_large.yml']
    stages = [0, 1, 2]
    for model_config in model_configs:
        for stage in stages:
            if model_config == 'cfg/multirange_model/main_donly.yml' and stage in [1,2]:
                continue
            if model_config == 'cfg/multirange_model/main_ponly.yml' and stage == 0:
                continue
            process = subprocess.Popen(['python','train_mr.py','--dataset_config',dataset_config,'--model_config',model_config,'--multirange_config',multirange_config,'--stage',str(stage)])
            process.wait()  # must be serialized