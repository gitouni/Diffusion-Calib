import subprocess

if __name__ == "__main__":
    python_path = '/home/ouni/miniconda3/envs/relposepp/bin/python'
    script_path = 'metrics.py'
    gt_dir = "cache/kitti_gt"
    log_file_fmt = "log/kitti/{method}_{suffix}.json"
    method_list = ['calibnet','lccnet','rggnet','lccraft_small','lccraft_large']
    iterative_list = ['single','naiter','lsd','nlsd']
    suffix_list = ['sig','iter','lsd','nlsd']
    pred_dirs = [
            [
                'experiments/kitti/naiter/calibnet/results/iterative_1_2025-02-02-09-08-44',
                'experiments/kitti/naiter/lccnet/results/iterative_1_2025-02-02-09-10-41',
                'experiments/kitti/naiter/rggnet/results/iterative_1_2025-02-02-09-12-40',
                'experiments/kitti/naiter/lccraft_small/results/iterative_1_2025-02-02-09-14-35',
                'experiments/kitti/naiter/lccraft_large/results/iterative_1_2025-02-02-09-17-20'
            ],
            [
                'experiments/kitti/naiter/calibnet/results/iterative_10_2025-02-02-09-21-25',
                'experiments/kitti/naiter/lccnet/results/iterative_10_2025-02-02-09-23-46',
                'experiments/kitti/naiter/rggnet/results/iterative_10_2025-02-02-09-28-33',
                'experiments/kitti/naiter/lccraft_small/results/iterative_10_2025-02-02-09-32-03',
                'experiments/kitti/naiter/lccraft_large/results/iterative_10_2025-02-02-09-52-23'
            ],
            [
                'experiments/kitti/lsd/calibnet/results/unipc_10_2025-02-02-08-04-07',
                'experiments/kitti/lsd/lccnet/results/unipc_10_2025-02-02-08-06-59',
                'experiments/kitti/lsd/rggnet/results/unipc_10_2025-02-02-08-12-18',
                'experiments/kitti/lsd/lccraft_small/results/unipc_10_2025-02-02-08-16-21',
                'experiments/kitti/lsd/lccraft_large/results/unipc_10_2025-02-02-08-37-54'
            ],
            [
                'experiments/kitti/nlsd/calibnet/results/nlsd_10_2025-02-02-10-22-07',
                'experiments/kitti/nlsd/lccnet/results/nlsd_10_2025-02-02-10-24-47',
                'experiments/kitti/nlsd/rggnet/results/nlsd_10_2025-02-02-10-29-55',
                'experiments/kitti/nlsd/lccraft_small/results/nlsd_10_2025-02-02-10-33-45',
                'experiments/kitti/nlsd/lccraft_large/results/nlsd_10_2025-02-02-10-54-22'
            ],
            # [
            #     'experiments/kitti/mr_3/calibnet/results/mr_3',
            #     'experiments/kitti/mr_3/lccnet/results/mr_3_2025-01-30-20-41-35',
            #     'experiments/kitti/mr_3/rggnet/results/mr_3_2025-01-30-20-43-42',
            #     'experiments/kitti/mr_3/lccraft_small/results/mr_3_2025-01-30-20-45-48',
            #     'experiments/kitti/mr_3/lccraft_large/results/mr_3_2025-01-30-20-53-24'
            # ]
        ]

    for method_i, method in enumerate(method_list):
        for mode_i, (iterative, suffix) in enumerate(zip(iterative_list, suffix_list)):
            print("Inference {} + {}".format(method, iterative))
            pred_dir = pred_dirs[mode_i][method_i]
            log_file = log_file_fmt.format(method=method, suffix=suffix)
            process = subprocess.Popen([python_path, script_path, '--pred_dir_root', pred_dir, '--gt_dir',gt_dir, '--log_file',log_file])
            process.wait()