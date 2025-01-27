import os
# from PIL import Image
# import importlib
# from datetime import datetime
import logging
import pandas as pd

# from . import tools as Util
from typing import Optional
import json
from time import strftime, localtime

def fmt_time(fmt:str="%Y-%m-%d-%H-%M-%S") -> str:
    return strftime(fmt, localtime())

def print_warning(s:str, **argv):
    print("\033[33;1m{}\033[0m".format(s), **argv)
    
class InfoLogger():
    """
    use logging to record log, only work on GPU 0 by judging global_rank
    """
    def __init__(self, opt):
        self.opt = opt
        self.phase = opt['phase']

        self.setup_logger(None, opt['path']['experiments_root'], opt['phase'], level=logging.INFO, screen=False)
        self.logger = logging.getLogger(opt['phase'])
        self.infologger_ftns = {'info', 'warning', 'debug'}

    def __getattr__(self, name):
        if name in self.infologger_ftns:
            print_info = getattr(self.logger, name, None)
            def wrapper(info, *args, **kwargs):
                print_info(info, *args, **kwargs)
            return wrapper
    
    @staticmethod
    def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
        """ set up logger """
        l = logging.getLogger(logger_name)
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        log_file = os.path.join(root, '{}.log'.format(phase))
        fh = logging.FileHandler(log_file, mode='a+')
        fh.setFormatter(formatter)
        l.setLevel(level)
        l.addHandler(fh)
        if screen:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            l.addHandler(sh)

class VisualWriter():
    """ 
    use tensorboard to record visuals, support 'add_scalar', 'add_scalars', 'add_image', 'add_images', etc. funtion.
    Also integrated with save results function.
    """
    def __init__(self, opt):
        self.result_dir = opt['path']['results']
        if 'save_suffix' in opt:
            self.suffix = opt['save_suffix']
        else:
            self.suffix = ".bmp"

        self.epoch = 0
        self.iters = 0
        self.phase = ''

    def set_iter(self, epoch, iters, phase='train'):
        self.phase = phase
        self.epoch = epoch
        self.iters = iters

    def save_res(self, results:dict):
        phase_path = os.path.join(self.result_dir, self.phase)
        os.makedirs(phase_path, exist_ok=True)
        result_path = os.path.join(phase_path, str(self.epoch))
        os.makedirs(result_path, exist_ok=True)
        for name, res in zip(results['name'], results['result']):
            json.dump(res, open(os.path.join(result_path, name + '.json'), 'w'), indent=2, sort_keys=False)
        

    # def save_images(self, results):
    #     phase_path = os.path.join(self.result_dir, self.phase)
    #     os.makedirs(phase_path, exist_ok=True)
    #     result_path = os.path.join(phase_path, str(self.epoch))
    #     os.makedirs(result_path, exist_ok=True)

    #     ''' get names and corresponding images from results[OrderedDict] '''
    #     try:
    #         names = results['name']
    #         outputs = Util.postprocess(results['result'])
    #         for i in range(len(names)): 
    #             Image.fromarray(outputs[i]).save(os.path.join(result_path, os.path.splitext(names[i])[0]+self.suffix))
    #     except:
    #         raise NotImplementedError('You must specify the context of name and result in save_current_results functions of model.')

    @property
    def result_path(self):
        return os.path.join(self.result_dir, self.phase, str(self.epoch))


class LogTracker:
    """
    record training numerical indicators.
    """
    def __init__(self, *keys, phase:Optional[str]=None):
        self.phase = phase
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        if self.phase is not None:
            return {'{}/{}'.format(self.phase, k):v for k, v in dict(self._data.average).items()}
        else:
            return dict(self._data.average)
