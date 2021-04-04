
import torch
import math

from from collections import defaultdict

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Trainer:
    """
    Intialize the trainer for the model

    Arguments
    ---------
    dataloader: torch.data object
        dataloader for specified dataset
    cfg_data: edict dictionary
        configuration file for training the model
    pwd: str, optional (default = )
        present working directory

    Returns
    -------
    tuple
        (x, y) if x, otherwise y

    """

    def __init__(self, dataloader, cfg_data, pwd):

        self.dataloader = dataloader
        self.cfg_data = cfg_data
        self.pwd = pwd

        self.train_record = {
            'best_mae': math.inf,
            'best_model_name': ''}
        self.timer = {
            'iter_time': Timer(),
            'train_time': Timer(),
            'val_time': Timer()}

        self.train_loader, self.val_loader = self.loader(*params)

        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH)
            self.net.load_state_dict(latest_state['net'])
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.scheduler.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch'] + 1
            ## need to unload the model details

        self.writer = logger()


    def forward(self):

        for epoch in range(self.epoch, cfg.MAX_EPOCH)
