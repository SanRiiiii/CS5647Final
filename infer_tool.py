import io
import logging
import os
import time
from pathlib import Path

import librosa
import numpy as np

import soundfile
import torch
import torchaudio

import sys


from utils import repeat_expand_2d, Volume_Extractor
from ComoSVC import load_model_vocoder
from slicer import cut, chunks2audio
from torch import Tensor

logging.getLogger('matplotlib').setLevel(logging.WARNING)


def format_wav(audio_path):
    if Path(audio_path).suffix == '.wav':
        return
    raw_audio, raw_sample_rate = librosa.load(audio_path, mono=True, sr=None)
    soundfile.write(Path(audio_path).with_suffix(".wav"), raw_audio, raw_sample_rate)


def get_end_file(dir_path, end):
    file_lists = []
    for root, dirs, files in os.walk(dir_path):
        files = [f for f in files if f[0] != '.']
        dirs[:] = [d for d in dirs if d[0] != '.']
        for f_file in files:
            if f_file.endswith(end):
                file_lists.append(os.path.join(root, f_file).replace("\\", "/"))
    return file_lists


def fill_a_to_b(a, b):
    if len(a) < len(b):
        for _ in range(0, len(b) - len(a)):
            a.append(a[0])

def mkdir(paths: list):
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

def pad_array(arr, target_length):
    current_length = arr.shape[0]
    if current_length >= target_length:
        return arr
    else:
        pad_width = target_length - current_length
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        padded_arr = np.pad(arr, (pad_left, pad_right), 'constant', constant_values=(0, 0))
        return padded_arr
    
def split_list_by_n(list_collection, n, pre=0):
    for i in range(0, len(list_collection), n):
        yield list_collection[i-pre if i-pre>=0 else i: i + n]


class F0FilterException(Exception):
    pass

class Svc(object):
    def __init__(self,
                 diffusion_model_path="logs/model_8000.pt",
                 diffusion_config_path="configs/diffusion.yaml",
                 total_steps=1,
                 dataset_path="./dataset"  # 新增：dataset路径，用于加载spk_embd
                 ):

        self.total_steps = total_steps
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diffusion_model, self.vocoder, self.diffusion_args = load_model_vocoder(diffusion_model_path, self.dev, config_path=diffusion_config_path, total_steps=self.total_steps)
        self.target_sample = self.diffusion_args.data.sampling_rate
        self.hop_size = self.diffusion_args.data.hop_length
        self.dtype = torch.float32
        self.speech_encoder = self.diffusion_args.data.encoder
        self.unit_interpolate_mode = self.diffusion_args.data.unit_interpolate_mode if self.diffusion_args.data.unit_interpolate_mode is not None else 'left'
        self.dataset_path = dataset_path  # 保存dataset路径

        # load hubert and model
        from Features import ContentVec768L12
        self.hubert_model = ContentVec768L12(device = self.dev)
        self.volume_extractor = Volume_Extractor(self.hop_size)
        
        # 缓存已加载的speaker embeddings
        self.spk_embd_cache = {}

    def load_speaker_embedding(self, speaker_name, random_select=True):
        """
        从dataset目录加载指定说话人的speaker embedding
        
        Args:
            speaker_name: 说话人名称（对应dataset下的文件夹名）
            random_select: 是否随机选择一个.spk.npy文件（默认True）
            
        Returns:
            spk_embd: [1, spk_embd_dim] 的tensor
        """
        # 构建路径：dataset/speaker_name/
        speaker_dir = os.path.join(self.dataset_path, speaker_name)
        if not os.path.exists(speaker_dir):
            raise FileNotFoundError(f"Speaker directory not found: {speaker_dir}")
        
        # 查找所有.spk.npy文件
        spk_files = [f for f in os.listdir(speaker_dir) if f.endswith('.spk.npy')]
        if not spk_files:
            raise FileNotFoundError(f"No .spk.npy file found in: {speaker_dir}")
        
        # 随机选择一个或选第一个
        if random_select:
            import random
            selected_file = random.choice(spk_files)
            print(f"🎲 Randomly selected from {len(spk_files)} speaker embeddings")
        else:
            selected_file = spk_files[0]
        
        spk_file_path = os.path.join(speaker_dir, selected_file)
        print(f"🎤 Loading speaker embedding: {speaker_name}/{selected_file}")
        
        # 加载speaker embedding
        spk_embd = np.load(spk_file_path)  # [spk_embd_dim]
        spk_embd = torch.from_numpy(spk_embd).float().unsqueeze(0).to(self.dev)  # [1, spk_embd_dim]
        
        print(f"✅ Loaded speaker embedding: shape={spk_embd.shape}")
        return spk_embd

    def get_unit_f0(self, wav, tran):

        if not hasattr(self,"f0_predictor_object") or self.f0_predictor_object is None:
            from Features import DioF0Predictor
            self.f0_predictor_object = DioF0Predictor(hop_length=self.hop_size,sampling_rate=self.target_sample) 
        f0, uv = self.f0_predictor_object.compute_f0_uv(wav)
        f0 = torch.FloatTensor(f0).to(self.dev)
        uv = torch.FloatTensor(uv).to(self.dev)

        f0 = f0 * 2 ** (tran / 12)
        f0 = f0.unsqueeze(0)
        uv = uv.unsqueeze(0)

        wav = torch.from_numpy(wav).to(self.dev)
        if not hasattr(self,"audio16k_resample_transform"):
            self.audio16k_resample_transform = torchaudio.transforms.Resample(self.target_sample, 16000).to(self.dev)
        wav16k = self.audio16k_resample_transform(wav[None,:])[0]
        
        c = self.hubert_model.encoder(wav16k)
        c = repeat_expand_2d(c.squeeze(0), f0.shape[1],self.unit_interpolate_mode)
        c = c.unsqueeze(0)
        return c, f0, uv
    
    def infer(self, speaker, tran, raw_path):
        """
        执行语音转换推理
        
        Args:
            speaker: 目标说话人名称（对应dataset下的文件夹名）
            tran: 音调偏移（半音）
            raw_path: 源音频路径
            
        Returns:
            audio: 转换后的音频
            audio_length: 音频长度
            n_frames: 帧数
        """
        torchaudio.set_audio_backend("soundfile")
        wav, sr = torchaudio.load(raw_path)
        if not hasattr(self,"audio_resample_transform") or self.audio16k_resample_transform.orig_freq != sr:
            self.audio_resample_transform = torchaudio.transforms.Resample(sr,self.target_sample)
        wav = self.audio_resample_transform(wav).numpy()[0]  # (100080,)

        # 1. 提取源音频的内容特征（units, f0, volume）
        c, f0, uv = self.get_unit_f0(wav, tran)
        n_frames = f0.size(1)
        c = c.to(self.dtype)
        f0 = f0.to(self.dtype)
        uv = uv.to(self.dtype)

        # 2. 加载目标说话人的speaker embedding
        spk_embd = self.load_speaker_embedding(speaker)  # [1, spk_embd_dim]

        with torch.no_grad():
            start = time.time()
            
            # 3. 提取源音频的音量
            audio = torch.FloatTensor(wav).to(self.dev)
            vol = self.volume_extractor.extract(audio[None,:])[None,:,None].to(self.dev)
            
            # 4. 调整维度
            f0 = f0[:,:,None]  # [1, T] -> [1, T, 1]
            c = c.transpose(-1,-2)  # [1, units, T] -> [1, T, units]
            
            # 5. 执行推理（使用预加载的spk_embd）
            # 模型返回 (audio_mel, attention_gate)
            audio_mel, attention_gate = self.diffusion_model(c, f0, vol, spk_embd=spk_embd, gt_spec=None, infer=True)
            
            # 6. 使用vocoder生成音频
            audio = self.vocoder.infer(audio_mel, f0).squeeze()
            
            use_time = time.time() - start
            print(f"⏱️  Inference time: {use_time:.2f}s")
            
        return audio, audio.shape[-1], n_frames

    def clear_empty(self):
        # clean up vram
        torch.cuda.empty_cache()


    def slice_inference(self,
                        raw_audio_path,
                        spk,
                        tran,
                        slice_db=-40, # -40
                        pad_seconds=0.5,
                        clip_seconds=0,
                        ):

        wav_path = Path(raw_audio_path).with_suffix('.wav')
        chunks = cut(wav_path, db_thresh=slice_db)
        audio_data, audio_sr = chunks2audio(wav_path, chunks) 
        per_size = int(clip_seconds*audio_sr)
        lg_size = 0
        global_frame = 0
        audio = []
        for (slice_tag, data) in audio_data:
            print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
            # padd
            length = int(np.ceil(len(data) / audio_sr * self.target_sample))
            if slice_tag:
                print('jump empty segment')
                _audio = np.zeros(length)
                audio.extend(list(pad_array(_audio, length)))
                global_frame += length // self.hop_size
                continue
            if per_size != 0:
                datas = split_list_by_n(data, per_size,lg_size)
            else:
                datas = [data]
            for k,dat in enumerate(datas):
                per_length = int(np.ceil(len(dat) / audio_sr * self.target_sample)) if clip_seconds!=0 else length
                if clip_seconds!=0: 
                    print(f'###=====segment clip start, {round(len(dat) / audio_sr, 3)}s======')
                # padd
                pad_len = int(audio_sr * pad_seconds)
                dat = np.concatenate([np.zeros([pad_len]), dat, np.zeros([pad_len])])
                raw_path = io.BytesIO()
                soundfile.write(raw_path, dat, audio_sr, format="wav")
                raw_path.seek(0)
                out_audio, out_sr, out_frame = self.infer(spk, tran, raw_path)
                global_frame += out_frame
                _audio = out_audio.cpu().numpy()
                pad_len = int(self.target_sample * pad_seconds)
                _audio = _audio[pad_len:-pad_len]
                _audio = pad_array(_audio, per_length)
                audio.extend(list(_audio))
        return np.array(audio)
