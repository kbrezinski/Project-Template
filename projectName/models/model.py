
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MyModel(nn.Module):
    def __init__(self, gpus):
        super(MyModel, self).__init__()

        self.gpus = gpus
        self.NN = net() ## import model

        if len(self.gpus) > 1:
            self.NN = torch.nn.DataParallel(
                self.NN, device_ids=self.gpus).to(device)
        else:
            self.NN = self.NN.to(device)
