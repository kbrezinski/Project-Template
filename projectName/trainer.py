
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

        self.epoch = 0
        self.net = None ## import model.to(device)

        self.train_record = {
            'best_mae': math.inf,
            'best_model_name': ''}
        self.timer = {
            'iter_time': Timer(),
            'train_time': Timer(),
            'val_time': Timer()}

        self.train_loader, self.val_loader = self.loader(*params)

        ## load previous state of model training
        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH)
            self.net.load_state_dict(latest_state['net'])
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.scheduler.load_state_dict(latest_state['scheduler'])
            self.epoch = latest_state['epoch'] + 1
            self.train_record = latest_state['train_record']
            ## need to unload the model details

        self.writer = logger()


    def forward(self):

        for epoch in range(self.epoch, cfg.MAX_EPOCH):

            self.epoch = epoch

            self.timer['train_time'].tic()
            self.train()
            self.timer['train_time'].toc()

            if epoch % cfg.VAL_FREQ == 0:
                self.timer['val_time'].tic()
                self.validate()
                self.timer['val_time'].toc(average=False)


    def train(self):
        ## in torch activates batch norm, dropout, etc.
        self.net.train()

        for i, data in enumerate(self.train_loader):
            self.timer['iter_time'].tic()
            ## x, y = data

            x = Variable(x).to(device)
            y = Variable(y).to(device)

            self.optimizer.zero_grad()
            y_pred, loss = self.net(x, y)
            loss.backward()

            clip_grad_norm_(self.net.parameters(), None) ## grad clip clip_grad_norm_
            self.optimizer.step()

            ## Print update metrics
            if (i + 1) % cfg.PRINT_FREQ == 0:
                self.writer.add_scalar('train_loss', loss.item(), i) ## or other counter
                self.timer['iter_time'].toc(average=False)
                print(
                    self.epoch,
                    i + 1,
                    loss.item(),
                    self.optimizer.param_groups[0]['lr'],
                    self.timer['iter_time'].diff()
                )

    def validate(self):

        ## sets self.train(False)
        self.net.eval()

        losses = AverageMeter()
        maes = AverageMeter()
        mses = AverageMeter()

        for vi, data in enumerate(self.val_loader):
            ## x, y = data

            with torch.no_grad():
                x = Variable(x).to(device)
                y = Variable(y).to(device)

                y_pred, loss = self.net.test(x, y)

                ## convert to numpy with .data.cpu().numpy()

                for i in range(x.shape[0]):

                    losses.update(loss)
                    #maes.update()
                    #mses.update()
                #mae = maes.avg
                #mse = np.sqrt(mses.avg)
                #loss = losses.avg

                self.writer.add_scalar('val_loss', loss, self.epoch + 1)
                self.writer.add_scalar('mae', mae, self.epoch + 1)
                self.writer.add_scalar('mse', mse, self.epoch + 1)

                self.train_record = update_model()
