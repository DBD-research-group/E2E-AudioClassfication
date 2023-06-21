import torch
import torch.nn.functional as F
import torchaudio
import os
import glob
import random
from datasets.audio_augs import AudioAugs


class ESCDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 mode,
                 segment_length,
                 sampling_rate,
                 transforms=None,
                 fold_id=None,
                 ):
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        fnames = glob.glob(root + "/**/*.wav")
        self._get_labels(fnames)
        if mode == 'train':
            fnames = [f for f in fnames if int(os.path.basename(f).split('-')[0]) != fold_id]
        elif mode == 'test':
            fnames = [f for f in fnames if int(os.path.basename(f).split('-')[0]) == fold_id]
        else:
            raise ValueError("wrong mode")
        self.audio_files = sorted(fnames)
        self.label2idx = dict(zip(self.labels, range(len(self.labels))))
        self.transforms = transforms
        # resample if needed
        test_sample_rate = torchaudio.load(self.audio_files[0])[1]
        if test_sample_rate != self.sampling_rate:
            self.audio_files = self._resample(self.audio_files, sampling_rate)

    def _resample(self, f_names, sampling_rate):
        resampled_fnames = []
        for f in f_names:
            audio, org_sample_rate = torchaudio.load(f)
            audio = torchaudio.transforms.Resample(orig_freq=org_sample_rate, new_freq=sampling_rate)(audio)
            torchaudio.save(f, audio, sampling_rate)
            resampled_fnames.append(f)
        return resampled_fnames

    def _get_labels(self, f_names):
        self.labels = sorted(list(set([f.split('/')[-2] for f in f_names])))

    def __getitem__(self, index):
        fname = self.audio_files[index]
        label = fname.split('/')[-2]
        label = self.label2idx[label]
        audio, sampling_rate = torchaudio.load(fname)
        audio.squeeze_()
        audio = 0.95 * (audio / audio.__abs__().max()).float()

        assert("sampling rate of the file is not as configured in dataset, will cause slow fetch {}".format(sampling_rate != self.sampling_rate))
        if audio.shape[0] >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data

        if self.transforms is not None:
            audio = AudioAugs(self.transforms, sampling_rate, p=0.5)(audio)

        return audio.unsqueeze(0), label

    def __len__(self):
        return len(self.audio_files)


if __name__ == "__main__":
    pass