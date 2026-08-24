from torch import nn

class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args
        self._name = 'reshape'

    def forward(self, x):
        return x.view(self.shape)

class BatchFlatten(nn.Module):
    def __init__(self):
        super(BatchFlatten, self).__init__()
        self._name = 'batch_flatten'

    def forward(self, x):
        return x.view(x.shape[0], -1)