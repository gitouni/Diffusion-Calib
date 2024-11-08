import subprocess

if __name__ == "__main__":
    python_file = 'metrics.py'
    pred_dir_fmt = "experiments/multirange/{method}/kitti/results/{filefolder}"
    gt_dir = "cache/kitti_gt"
    log_file_fmt = "log/multirange/{savefile}.json"
    method_list = ['lccraft_small','lccraft_large']
    iterative_list = ['mr_5']
    suffix_list = ['_mr5']
    for method in method_list:
        for iterative, suffix in zip(iterative_list, suffix_list):
            print("Inference {} + {}".format(method, iterative))
            pred_dir = pred_dir_fmt.format(method=method, filefolder=iterative)
            log_file = log_file_fmt.format(savefile=method+suffix)
            process = subprocess.Popen(['python',python_file,'--pred_dir_root',pred_dir, '--gt_dir',gt_dir, '--log_file',log_file])
            process.wait()