import os
from collections import OrderedDict
import yaml
from pathlib import Path
from datetime import datetime
from functools import partial
import importlib
from types  import FunctionType
import shutil


def init_obj(opt, logger, *args, default_file_name='default file', given_module=None, init_type='Network', **modify_kwargs):
    """
    finds a function handle with the name given as 'name' in config,
    and returns the instance initialized with corresponding args.
    """ 
    if opt is None or len(opt)<1:
        logger.info('Option is None when initialize {}'.format(init_type))
        return None
    
    ''' default format is dict with name key '''
    if isinstance(opt, str):
        opt = {'name': opt}
        logger.warning('Config is a str, converts to a dict {}'.format(opt))

    name = opt['name']
    ''' name can be list, indicates the file and class name of function '''
    if isinstance(name, list):
        file_name, class_name = name[0], name[1]
    else:
        file_name, class_name = default_file_name, name
    try:
        if given_module is not None:
            module = given_module
        else:
            module = importlib.import_module(file_name)
        
        attr = getattr(module, class_name)
        kwargs = opt.get('args', {})
        kwargs.update(modify_kwargs)
        ''' import class or function with args '''
        if isinstance(attr, type): 
            ret = attr(*args, **kwargs)
            ret.__name__  = ret.__class__.__name__
        elif isinstance(attr, FunctionType): 
            ret = partial(attr, *args, **kwargs)
            ret.__name__  = attr.__name__
            # ret = attr
        logger.info('{} [{:s}() form {:s}] is created.'.format(init_type, class_name, file_name))
    except Exception as e:
        raise RuntimeError('{} [{:s}() form {:s}] exception: {}.'.format(init_type, class_name, file_name, e))
    return ret


def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)

def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')


def write_yaml(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        yaml.dump(content, handle, yaml.SafeDumper, indent=2, sort_keys=False)

class NoneDict(dict):
    def __missing__(self, key):
        return None

def dict_to_nonedict(opt):
    """ convert to NoneDict, which return None for missing key. """
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt

def dict2str(opt, indent_l=1):
    """ dict to string for logger """
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg

def parse(args):
    opt = yaml.load(open(args.config,'r'), yaml.SafeLoader)

    ''' replace the config context using args '''
    opt['phase'] = args.phase
    if args.gpu_ids is not None:
        opt['gpu_ids'] = [int(id) for id in args.gpu_ids.split(',')]
    if args.batch is not None:
        opt['datasets'][opt['phase']]['dataloader']['args']['batch_size'] = args.batch
 
    ''' set cuda environment '''
    assert len(opt['gpu_ids']) == 1, "this code does not support multiple gpus"

    ''' update name '''
    if args.debug:
        opt['name'] = 'debug_{}'.format(opt['name'])
    elif opt['finetune_norm']:
        opt['name'] = 'finetune_{}'.format(opt['name'])
    else:
        opt['name'] = '{}_{}'.format(opt['phase'], opt['name'])

    ''' set log directory '''
    experiments_root = os.path.join(opt['path']['base_dir'], '{}_{}'.format(opt['name'], get_timestamp()))
    mkdirs(experiments_root)

    ''' save json '''
    write_yaml(opt, '{}/config.yml'.format(experiments_root))

    ''' change folder relative hierarchy '''
    opt['path']['experiments_root'] = experiments_root
    for key, path in opt['path'].items():
        if 'resume' not in key and 'base' not in key and 'root' not in key:
            opt['path'][key] = os.path.join(experiments_root, path)
            mkdirs(opt['path'][key])

    ''' debug mode '''
    if 'debug' in opt['name']:
        opt['train'].update(opt['debug'])

    ''' code backup ''' 
    # for name in os.listdir('.'):
    #     if name in ['models', 'core', 'utils']:
    #         shutil.copytree(name, os.path.join(opt['path']['code'], name), ignore=shutil.ignore_patterns("*.pyc", "__pycache__"))
    #     if '.py' in name or '.sh' in name:
    #         shutil.copy(name, opt['path']['code'])
    return dict_to_nonedict(opt)





