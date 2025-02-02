import subprocess

if __name__ == "__main__":
    python_path = '/home/ouni/miniconda3/envs/relposepp/bin/python'
    # LSD
    script_path = 'test.py'
    configs = ['experiments/kitti/lsd/calibnet/log/kitti_lsd_calibnet.yml',
                'experiments/kitti/lsd/lccnet/log/kitti_lsd_lccnet.yml',
                'experiments/kitti/lsd/rggnet/log/kitti_lsd_rggnet.yml',
                'experiments/kitti/lsd/lccraft_small/log/kitti_lsd_lccraft_small.yml',
                'experiments/kitti/lsd/lccraft_large/log/kitti_lsd_lccraft_large.yml']
    diffusion_args = ['--model_type','diffusion']
    for cfg in configs:
        process = subprocess.Popen([python_path, script_path, '--config',cfg, *diffusion_args])
        process.wait()  # must be serialized
    # Single
    configs = ['experiments/kitti/naiter/calibnet/log/kitti_naiter_calibnet.yml',
                'experiments/kitti/naiter/lccnet/log/kitti_naiter_lccnet.yml',
                'experiments/kitti/naiter/rggnet/log/kitti_naiter_rggnet.yml',
                'experiments/kitti/naiter/lccraft_small/log/kitti_naiter_lccraft_small.yml',
                'experiments/kitti/naiter/lccraft_large/log/kitti_naiter_lccraft_large.yml']
    iterative_args = ['--model_type','iterative','--iters','1']
    for cfg in configs:
        process = subprocess.Popen([python_path, script_path, '--config',cfg, *iterative_args])
        process.wait()  # must be serialized
    # NaIter
    iterative_args = ['--model_type','iterative','--iters','10']
    for cfg in configs:
        process = subprocess.Popen([python_path, script_path, '--config',cfg, *iterative_args])
        process.wait()  # must be serialized
    # NLSD
    script_path = 'test_nlsd.py'
    configs = ['experiments/kitti/nlsd/calibnet/log/kitti_nlsd_calibnet.yml',
                'experiments/kitti/nlsd/lccnet/log/kitti_nlsd_lccnet.yml',
                'experiments/kitti/nlsd/rggnet/log/kitti_nlsd_rggnet.yml',
                'experiments/kitti/nlsd/lccraft_small/log/kitti_nlsd_lccraft_small.yml',
                'experiments/kitti/nlsd/lccraft_large/log/kitti_nlsd_lccraft_large.yml']
    for cfg in configs:
        process = subprocess.Popen([python_path, script_path, '--config',cfg])
        process.wait()  # must be serialized
    # MR
    # script_path = 'test_mr.py'
    # # configs = ['experiments/kitti/mr_3/calibnet/log/kitti_mr_3_calibnet.yml',
    # #             'experiments/kitti/mr_3/lccnet/log/kitti_mr_3_lccnet.yml',
    # #             'experiments/kitti/mr_3/rggnet/log/kitti_mr_3_rggnet.yml',
    # #             'experiments/kitti/mr_3/lccraft_small/log/kitti_mr_3_lccraft_small.yml',
    # #             'experiments/kitti/mr_3/lccraft_large/log/kitti_mr_3_lccraft_large.yml']
    # configs = [
    #             'experiments/kitti/mr_3/lccnet/log/kitti_mr_3_lccnet.yml',
    #             'experiments/kitti/mr_3/rggnet/log/kitti_mr_3_rggnet.yml',
    #             'experiments/kitti/mr_3/lccraft_small/log/kitti_mr_3_lccraft_small.yml',
    #             'experiments/kitti/mr_3/lccraft_large/log/kitti_mr_3_lccraft_large.yml']
    # for cfg in configs:
    #     process = subprocess.Popen([python_path, script_path, '--config',cfg])
    #     process.wait()  # must be serialized
    
    