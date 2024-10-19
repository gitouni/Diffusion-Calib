import argparse
import torch
import os
import sys
sys.path.append(os.path.dirname(__file__))
from core.logger import VisualWriter, InfoLogger
import core.parser as Parser
import core.tools as Util
from models import define_dataloader
from models import create_model, define_network, define_loss, define_metric
from models.trainer import Trainer


def main_worker(opt):
    '''set seed and and cuDNN environment '''
    torch.backends.cudnn.enabled = True
    # warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])

    ''' set logger '''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt)  
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    '''set networks and dataset'''
    phase_loader, val_loader = define_dataloader(phase_logger, opt) # val_loader is None if phase is test.
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]
    if opt['phase'] == 'test' and opt['skip_iter'] != 0:
        for _ in range(opt['skip_iter']):
            phase_loader.__iter__()
    ''' set metrics, loss, optimizer and  schedulers '''
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    model:Trainer = create_model(
        opt = opt,
        networks = networks,
        phase_loader = phase_loader,
        val_loader = val_loader,
        losses = losses,
        metrics = metrics,
        logger = phase_logger,
        writer = phase_writer
    )

    phase_logger.info('Begin model {}.'.format(opt['phase']))
    try:
        if opt['phase'] == 'train':
            model.train()
        elif opt['phase'] == 'test':
            model.test()
        else:
            raise NotImplementedError("phase {} not implemented.".format(opt['phase']))
    finally:
        pass
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='cfg/kitti.yml', help='config file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train','test'], help='Run train, test', default='train')
    parser.add_argument('-skip','--skip_iter',type=int,default=0)
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)

    ''' parser configs '''
    args = parser.parse_args()
    opt = Parser.parse(args)
    
    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    ''' use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training'''
    # [Todo]: multi GPU on multi machine
    opt['world_size'] = 1 
    main_worker(opt)