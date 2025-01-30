import subprocess

if __name__ == "__main__":
    python_file = 'metrics.py'
    gt_dir = "cache/kitti_gt"
    log_file_fmt = "log/large/{method}_{suffix}.json"
    method_list = ['calibnet','rggnet','lccnet','lccraft_small','lccraft_large','main']
    iterative_list = ['single','naiter','lsd','nlsd']
    suffix_list = ['_sig','_iter','_lsd','_nlsd']
    pred_dirs = [
        []
        ]

    for j, method in enumerate(method_list):
        for i, (iterative, suffix) in enumerate(zip(iterative_list, suffix_list)):
            print("Inference {} + {}".format(method, iterative))
            pred_dir = pred_dir_fmt.format(method=method, filefolder=iterative)
            log_file = log_file_fmt.format(savefile=method+suffix)
            process = subprocess.Popen(['python',python_file,'--pred_dir_root',pred_dir, '--gt_dir',gt_dir, '--log_file',log_file])
            process.wait()