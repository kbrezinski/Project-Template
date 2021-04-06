
import time
import os

def update_model():

    snapshot_name = "{} {} {}".format(epoch + 1, mae, mse)

    if mae < train_record['best_mae']:
        train_record['best_model_name'] = snapshot_name
        saved_weights = net.state_dict()
        torch.save(saved_weights,
                    path + exp + snapshot_name + ".pth")

    ## adjust train record for best metrics

    latest_state = {
        'train_record' : train_record,
        'net' : net.state_dict,
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict(),
        'epoch' : epoch,
        'exp_path' : exp_path,
        'exp_name' : exp_name
        }

    torch.save(latest_state, path + exp + "latest_state.pth")

    return train_record


def logger(exp_path, exp_name):
    from tensorboardx import SummaryWriter

    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    writer = SummaryWriter(exp_path + "/" + exp_name)

    return writer

class Timer:

    def __init__(self):
        self.start_time = 0.
        self.total_time = 0.
        self.calls = 0
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count
