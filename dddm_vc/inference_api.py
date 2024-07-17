"""Copyright: Nabarun Goswami (2024)."""

import os
import torch
import torch.nn as nn
import argparse
import numpy as np
from huggingface_hub import snapshot_download
from torch.nn import functional as F
from scipy.io.wavfile import write
import torchaudio
from transformers.modeling_utils import ModuleUtilsMixin

import utils
from dddm_vc.data_loader import MelSpectrogramFixed

import amfm_decompy.basic_tools as basic
import amfm_decompy.pYAAPT as pYAAPT
from dddm_vc.model.vc_dddm_mixup import Wav2vec2, DDDM
from dddm_vc.vocoder.hifigan import HiFi
from dddm_vc.model_f0_vqvae import Quantizer


seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def load_audio(path):
    audio, sr = torchaudio.load(path)
    audio = audio[:1]
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000, resampling_method="kaiser_window")

    p = (audio.shape[-1] // 1280 + 1) * 1280 - audio.shape[-1]
    audio = torch.nn.functional.pad(audio, (0, p))

    return audio


def save_audio(wav, out_file, syn_sr=16000):
    wav = (wav.squeeze() / wav.abs().max() * 0.999 * 32767.0).cpu().numpy().astype('int16')
    write(out_file, syn_sr, wav)


def get_yaapt_f0(audio, sr=16000, interp=False):
    to_pad = int(20.0 / 1000 * sr) // 2
    f0s = []
    for y in audio.astype(np.float64):
        y_pad = np.pad(y.squeeze(), (to_pad, to_pad), "constant", constant_values=0)
        pitch = pYAAPT.yaapt(basic.SignalObj(y_pad, sr),
                             **{'frame_length': 20.0, 'frame_space': 5.0, 'nccf_thresh1': 0.25, 'tda_frame_length': 25.0})
        f0s.append(pitch.samp_interp[None, None, :] if interp else pitch.samp_values[None, None, :])

    return np.vstack(f0s)


class DDDMVCInferenceModel(nn.Module, ModuleUtilsMixin):
    def __init__(self,
                 hf_repo="subatomicseer/dddm_vc_checkpoints",
                 time_step=6,
                 ):
        super().__init__()

        self.time_step = time_step

        ckpts_dir = snapshot_download(hf_repo)

        ckpt_model = os.path.join(ckpts_dir, 'ckpt', "model_base.pth")
        ckpt_f0_vqvae = os.path.join(ckpts_dir, 'f0_vqvae', "f0_vqvae.pth")
        ckpt_voc = os.path.join(ckpts_dir, 'vocoder', "voc_ckpt.pth")

        config = os.path.join(os.path.split(ckpt_model)[0], 'config.json')
        hps = utils.get_hparams_from_file(config)

        self.mel_fn = MelSpectrogramFixed(
            sample_rate=hps.data.sampling_rate,
            n_fft=hps.data.filter_length,
            win_length=hps.data.win_length,
            hop_length=hps.data.hop_length,
            f_min=hps.data.mel_fmin,
            f_max=hps.data.mel_fmax,
            n_mels=hps.data.n_mel_channels,
            window_fn=torch.hann_window
        )

        # Load pre-trained w2v (XLS-R)
        self.w2v = Wav2vec2()

        # Load model
        self.f0_quantizer = Quantizer(hps)
        utils.load_checkpoint(ckpt_f0_vqvae, self.f0_quantizer)
        self.f0_quantizer.eval()

        self.model = DDDM(hps.data.n_mel_channels, hps.diffusion.spk_dim,
                     hps.diffusion.dec_dim, hps.diffusion.beta_min, hps.diffusion.beta_max, hps)
        utils.load_checkpoint(ckpt_model, self.model, None)
        self.model.eval()

        # Load vocoder
        self.net_v = HiFi(hps.data.n_mel_channels, hps.train.segment_size // hps.data.hop_length, **hps.model)
        utils.load_checkpoint(ckpt_voc, self.net_v, None)
        self.net_v.eval().dec.remove_weight_norm()

    def vc_to_file(self, audio, save_path, speaker_prompt):
        audio = load_audio(audio)

        src_mel = self.mel_fn(audio.to(self.device))
        src_length = torch.LongTensor([src_mel.size(-1)]).to(self.device)
        w2v_x = self.w2v(F.pad(audio, (40, 40), "reflect").to(self.device))

        try:
            f0 = get_yaapt_f0(audio.numpy())
        except:
            f0 = np.zeros((1, audio.shape[-1] // 80), dtype=np.float32)

        f0_x = f0.copy()
        f0_x = torch.log(torch.FloatTensor(f0_x + 1)).to(self.device)
        ii = f0 != 0
        f0[ii] = (f0[ii] - f0[ii].mean()) / f0[ii].std()
        f0 = torch.FloatTensor(f0).to(self.device)
        f0_code = self.f0_quantizer.code_extraction(f0)

        # trg_name = os.path.splitext(os.path.basename(a.trg_path))[0]
        trg_audio = load_audio(speaker_prompt)

        trg_mel = self.mel_fn(trg_audio.to(self.device))
        trg_length = torch.LongTensor([trg_mel.size(-1)]).to(self.device)

        with torch.no_grad():
            c = self.model.vc(src_mel, w2v_x, f0_code, src_length, trg_mel, trg_length, n_timesteps=self.time_step, mode='ml')
            converted_audio = self.net_v(c)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_audio(converted_audio, save_path)
