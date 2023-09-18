import torch

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    with torch.no_grad():
        correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].view(-1).float().sum(0) * 100. / batch_size for k in topk]

class AugBasic:
    def __init__(self, fs):
        super().__init__()
        self.fs = fs
        self.fft_params = {}
        if fs == 22050:
            self.fft_params['win_len'] = [512, 1024, 2048]
            self.fft_params['hop_len'] = [128, 256, 1024]
            self.fft_params['n_fft'] = [512, 1024, 2048]
        elif fs == 16000:
            self.fft_params['win_len'] = [256, 512, 1024]
            self.fft_params['hop_len'] = [256 // 4, 512 // 4, 1024 // 4]
            self.fft_params['n_fft'] = [256, 512, 1024]
        elif fs == 8000:
            self.fft_params['win_len'] = [128, 256, 512]
            self.fft_params['hop_len'] = [32, 64, 128]
            self.fft_params['n_fft'] = [128, 256, 512]
        else:
            raise ValueError