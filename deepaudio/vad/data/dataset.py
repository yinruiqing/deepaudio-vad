from typing import List, Union
from omegaconf import DictConfig

import numpy as np

from torch import Tensor
from torch.utils.data import Dataset

from pyannote.core import Segment, Timeline, SlidingWindow

from deepaudio.vad.data.audio_io.with_torchaudio import Audio
from deepaudio.vad.data.augmentation.noise import Noise, NoiseReverb, Reverb
from deepaudio.vad.data.augmentation.spec_augment import SpecAugment
from deepaudio.vad.data.feature import AUDIO_FEATURE_TRANSFORM_REGISTRY


class Seq2SeqDataset(Dataset):
    NONE_AUGMENT = 0
    NOISE_AUGMENT = 1
    REVERB_AUGMENT = 2
    NOISE_REVERB_AUGMENT = 3
    SPEC_AUGMENT = 4

    def __init__(
            self,
            configs: DictConfig,
            utts: List,
    ) -> None:
        super(Seq2SeqDataset, self).__init__()
        self.configs = configs
        self.utts = utts
        self.audio = Audio()
        self.feature_extractor = AUDIO_FEATURE_TRANSFORM_REGISTRY[configs.feature.name](configs)
        self.augmentations = [self.NONE_AUGMENT]
        weights = [1]
        if self.configs.augment.apply_noise_augment:
            self._noise_augmentor = Noise(configs)
            self.augmentations.append(self.NOISE_AUGMENT)
            weights.append(self.configs.augment.noise_augment_weight)

        if self.configs.augment.apply_reverb_augment:
            self._reverb_augmentor = Reverb(configs)
            self.augmentations.append(self.REVERB_AUGMENT)
            weights.append(self.configs.augment.reverb_augment_weight)
        if self.configs.augment.apply_noise_reverb_augment:
            self._noise_reverb_augmentor = NoiseReverb(configs)
            self.augmentations.append(self.NOISE_REVERB_AUGMENT)
            weights.append(self.configs.augment.noise_reverb_augment_weight)
        if self.configs.augment.apply_spec_augment:
            self._spec_augmentor = SpecAugment(configs)
            self.augmentations.append(self.SPEC_AUGMENT)
            weights.append(self.configs.augment.specaugment_weight)
        self.augmentations_prob = [float(i) / sum(weights) for i in weights]

    def _parse_audio(self, audio_path: str, augment: int = None, vad: Union[Segment, Timeline] = None) -> Tensor:
        if vad is not None:
            waveform, _ = self.audio.crop(audio_path, vad)
        else:
            waveform, _ = self.audio(audio_path)
        if augment == self.NOISE_AUGMENT:
            waveform = self._noise_augmentor(waveform)
        if augment == self.REVERB_AUGMENT:
            waveform = self._reverb_augmentor(waveform)
        if augment == self.NOISE_REVERB_AUGMENT:
            waveform = self._noise_reverb_augmentor(waveform)
        feature = self.feature_extractor(waveform)
        if augment == self.SPEC_AUGMENT:
            feature = self._spec_augmentor(feature)
        return feature.squeeze(0)

    def __getitem__(self, idx):
        wav, speech, chunk = self.utts[idx]
        augment = np.random.choice(self.augmentations, p=self.augmentations_prob)
        feature = self._parse_audio(wav, augment, chunk)
        num_frames, _ = feature.shape # TODO infer the num_frames by model
        y = np.zeros((num_frames, 1), dtype=np.int8)
        frames = SlidingWindow(
            start=chunk.start,
            duration=chunk.duration / num_frames,
            step=chunk.duration / num_frames,
        )
        for start, stop in frames.crop(speech, mode="center", return_ranges=True):
            y[start:stop] = 1
        return feature, y

    def __len__(self):
        return len(self.utts)


class Chunk2labelDataset(Seq2SeqDataset):
    def __getitem__(self, idx):
        wav, ann, chunk = self.utts[idx]
        augment = np.random.choice(self.augmentations, p=self.augmentations_prob)
        feature = self._parse_audio(wav, augment, chunk)
        speech = ann.get_timeline().support()
        speech_ratio = speech.duration() / chunk.duration
        y = 0 if speech_ratio < 0.5 else 1
        return feature, y
