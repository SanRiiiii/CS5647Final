import os
import random

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import repeat_expand_2d


def traverse_dir(
        root_dir,
        extensions,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True):

    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any([file.endswith(f".{ext}") for ext in extensions]):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list
                
                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue
                
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


def get_data_loaders(args, whole_audio=False, use_domain_labels=False):
    """
    获取数据加载器
    
    Args:
        args: 配置参数
        whole_audio: 是否使用完整音频
        use_domain_labels: 是否使用域标签
    """
    data_train = AudioDataset(
        filelists=args.data.training_files,
        waveform_sec=args.data.duration,
        hop_size=args.data.hop_length,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=whole_audio,
        extensions=args.data.extensions,
        device=args.train.cache_device,
        fp16=args.train.cache_fp16,
        unit_interpolate_mode=args.data.unit_interpolate_mode,
        use_aug=True,
    )
    
    # 使用标准采样方式
    loader_train = torch.utils.data.DataLoader(
        data_train,
        batch_sampler=None,
        batch_size=args.train.batch_size if not whole_audio else 1,
        sampler=None,
        shuffle=True,  # 随机打乱
        num_workers=args.train.num_workers if args.train.cache_device=='cpu' else 0,
        persistent_workers=(args.train.num_workers > 0) if args.train.cache_device=='cpu' else False,
        pin_memory=True if args.train.cache_device=='cpu' else False
    )
    
    data_valid = AudioDataset(
        filelists=args.data.validation_files,
        waveform_sec=args.data.duration,
        hop_size=args.data.hop_length,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=True,
        extensions=args.data.extensions,
        device=args.train.cache_device,
        fp16=args.train.cache_fp16,
        unit_interpolate_mode=args.data.unit_interpolate_mode,
    )
    
    loader_valid = torch.utils.data.DataLoader(
        data_valid,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return loader_train, loader_valid 


class AudioDataset(Dataset):
    def __init__(
        self,
        filelists,
        waveform_sec,
        hop_size,
        sample_rate,
        load_all_data=True,
        whole_audio=False,
        extensions=['wav'],
        device='cpu',
        fp16=False,
        use_aug=False,
        unit_interpolate_mode = 'left',
    ):
        super().__init__()
        
        self.waveform_sec = waveform_sec
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.filelists = filelists
        self.whole_audio = whole_audio
        self.use_aug = use_aug
        self.data_buffer={}
        self.pitch_aug_dict = {}
        self.unit_interpolate_mode = unit_interpolate_mode
        
        # np.load(os.path.join(self.path_root, 'pitch_aug_dict.npy'), allow_pickle=True).item()
        if load_all_data:
            print('Load all the data filelists:', filelists)
        else:
            print('Load the f0, volume data filelists:', filelists)
        with open(filelists,"r") as f:
            all_paths = f.read().splitlines()
        
        # 用于存储有效的路径（跳过缺失文件的样本）
        self.paths = []
        
        for name_ext in tqdm(all_paths, total=len(all_paths)):
            try:
                path_audio = name_ext
                if not os.path.exists(path_audio):
                    print(f"Warning: Audio file not found, skipping: {path_audio}")
                    continue
                    
                duration = librosa.get_duration(filename = path_audio, sr = self.sample_rate)
                if duration <= 0:
                    print(f"Warning: Invalid duration ({duration}), skipping: {path_audio}")
                    continue
                
                # Load F0
                path_f0 = name_ext + ".f0.npy"
                if not os.path.exists(path_f0):
                    print(f"Warning: F0 file not found, skipping: {path_f0}")
                    continue
                f0, _ = np.load(path_f0, allow_pickle=True)
                f0 = torch.from_numpy(np.array(f0, dtype=float)).float().unsqueeze(-1).to(device)
                
                # Load F0 distribution labels
                path_f0_dist = name_ext + ".dist.npy"
                if not os.path.exists(path_f0_dist):
                    print(f"Warning: F0 distribution file not found, skipping: {path_f0_dist}")
                    continue
                f0_dist_data = np.load(path_f0_dist)
                f0_dist = torch.tensor(f0_dist_data, dtype=torch.long).to(device)  # [5]
                
                # Load volume
                path_volume = name_ext + ".vol.npy"
                if not os.path.exists(path_volume):
                    print(f"Warning: Volume file not found, skipping: {path_volume}")
                    continue
                volume_data = np.load(path_volume)
                volume = torch.from_numpy(volume_data).float().unsqueeze(-1).to(device)
                
                # Load augmented volume
                path_augvol = name_ext + ".aug_vol.npy"
                if not os.path.exists(path_augvol):
                    print(f"Warning: Augmented volume file not found, skipping: {path_augvol}")
                    continue
                aug_vol_data = np.load(path_augvol)
                aug_vol = torch.from_numpy(aug_vol_data).float().unsqueeze(-1).to(device)
                            
                # Load domain label
                path_domain = name_ext.replace('.wav', '') + ".domain.npy"
                if not os.path.exists(path_domain):
                    print(f"Warning: Domain label file not found, skipping: {path_domain}")
                    continue
                domain_data = np.load(path_domain)
                domain_label = torch.from_numpy(domain_data).to(device)
                
                # Load speaker embedding
                path_spk_embd = name_ext.replace('.wav', '.spk.npy')
                if not os.path.exists(path_spk_embd):
                    print(f"Warning: Speaker embedding file not found, skipping: {path_spk_embd}")
                    continue
                spk_embd_data = np.load(path_spk_embd)
                spk_embd = torch.from_numpy(spk_embd_data).float().to(device)

                if load_all_data:
                    # Load mel spectrogram
                    path_mel = name_ext + ".mel.npy"
                    if not os.path.exists(path_mel):
                        print(f"Warning: Mel spectrogram file not found, skipping: {path_mel}")
                        continue
                    mel_data = np.load(path_mel)
                    mel = torch.from_numpy(mel_data).to(device)
                    
                    # Load augmented mel spectrogram
                    path_augmel = name_ext + ".aug_mel.npy"
                    if not os.path.exists(path_augmel):
                        print(f"Warning: Augmented mel spectrogram file not found, skipping: {path_augmel}")
                        continue
                    aug_mel, keyshift = np.load(path_augmel, allow_pickle=True)
                    aug_mel = np.array(aug_mel, dtype=float)
                    aug_mel = torch.from_numpy(aug_mel).to(device)
                    self.pitch_aug_dict[name_ext] = keyshift

                    # Load units
                    path_units = name_ext + ".soft.pt"
                    if not os.path.exists(path_units):
                        print(f"Warning: Units file not found, skipping: {path_units}")
                        continue
                    
                    units_data = torch.load(path_units)
                    units = units_data.to(device)
                    
                    # 修复：如果第一个维度是1，去掉它
                    if units.shape[0] == 1:
                        units = units[0]
                    
                    units = repeat_expand_2d(units, f0.size(0), unit_interpolate_mode).transpose(0, 1)
                
                    if fp16:
                        mel = mel.half()
                        aug_mel = aug_mel.half()
                        units = units.half()
                        # f0_dist是long类型（类别标签），不需要转换为half
                        
                    buffer_data = {
                            'duration': duration,
                            'mel': mel,
                            'aug_mel': aug_mel,
                            'units': units,
                            'f0': f0,
                            'f0_dist': f0_dist,
                            'volume': volume,
                            'aug_vol': aug_vol,
                            'domain_label': domain_label,
                            'spk_embd': spk_embd
                            }
                    
                    self.data_buffer[name_ext] = buffer_data
                    self.paths.append(name_ext)  # 成功加载，添加到有效路径列表
                else:
                    # Load augmented mel spectrogram for keyshift info
                    path_augmel = name_ext + ".aug_mel.npy"               
                    if not os.path.exists(path_augmel):
                        print(f"Warning: Augmented mel spectrogram file not found, skipping: {path_augmel}")
                        continue
                    aug_mel, keyshift = np.load(path_augmel, allow_pickle=True)
                    self.pitch_aug_dict[name_ext] = keyshift
                    
                    buffer_data = {
                            'duration': duration,
                            'f0': f0,
                            'f0_dist': f0_dist,
                            'volume': volume,
                            'aug_vol': aug_vol,
                            'domain_label': domain_label,
                            'spk_embd': spk_embd
                            }
                    
                    self.data_buffer[name_ext] = buffer_data
                    self.paths.append(name_ext)  # 成功加载，添加到有效路径列表
                    
            except Exception as e:
                print(f"Error loading sample {name_ext}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.paths)} samples out of {len(all_paths)} total samples.")
    
    def __getitem__(self, file_idx):
        name_ext = self.paths[file_idx]
        data_buffer = self.data_buffer[name_ext]
        
        # check duration. if too short, then skip
        if data_buffer['duration'] < (self.waveform_sec + 0.1):
            return self.__getitem__((file_idx + 1) % len(self.paths))
        
        # 检查整个音频的F0是否全为0（没有有声段）
        f0 = data_buffer.get('f0')
        if f0 is not None and (f0 <= 0).all():
            # 整个音频没有有声段，跳过这个音频
            return self.__getitem__((file_idx + 1) % len(self.paths))
            
        # get item
        return self.get_data(name_ext, data_buffer, file_idx)

    def get_data(self, name_ext, data_buffer, file_idx=None):
        name = os.path.splitext(name_ext)[0]
        frame_resolution = self.hop_size / self.sample_rate
        duration = data_buffer['duration']
        waveform_sec = duration if self.whole_audio else self.waveform_sec
        
        # load audio
        idx_from = 0 if self.whole_audio else random.uniform(0, duration - waveform_sec - 0.1)
        start_frame = int(idx_from / frame_resolution)
        units_frame_len = int(waveform_sec / frame_resolution)
        aug_flag = random.choice([True, False]) and self.use_aug
        '''
        audio = data_buffer.get('audio')
        if audio is None:
            path_audio = os.path.join(self.path_root, 'audio', name) + '.wav'
            audio, sr = librosa.load(
                    path_audio, 
                    sr = self.sample_rate, 
                    offset = start_frame * frame_resolution,
                    duration = waveform_sec)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            # clip audio into N seconds
            audio = audio[ : audio.shape[-1] // self.hop_size * self.hop_size]       
            audio = torch.from_numpy(audio).float()
        else:
            audio = audio[start_frame * self.hop_size : (start_frame + units_frame_len) * self.hop_size]
        '''
        # load mel
        mel_key = 'aug_mel' if aug_flag else 'mel'
        mel = data_buffer.get(mel_key)
        if mel is None:
            mel_path = name_ext + ".mel.npy"
            if not os.path.exists(mel_path):
                print(f"Warning: Mel file not found during dynamic loading: {mel_path}")
                return self.__getitem__((file_idx + 1) % len(self.paths))
            mel_data = np.load(mel_path)
            mel = mel_data[start_frame : start_frame + units_frame_len]
            mel = torch.from_numpy(mel).float() 
        else:
            mel = mel[start_frame : start_frame + units_frame_len]
            
        # load f0
        f0 = data_buffer.get('f0')
        aug_shift = 0
        if aug_flag:
            aug_shift = self.pitch_aug_dict[name_ext]
        f0_frames = 2 ** (aug_shift / 12) * f0[start_frame : start_frame + units_frame_len]
        
        # load units
        units = data_buffer.get('units')
        if units is None:
            path_units = name_ext + ".soft.pt"
            if not os.path.exists(path_units):
                print(f"Warning: Units file not found during dynamic loading: {path_units}")
                return self.__getitem__((file_idx + 1) % len(self.paths))
            units_data = torch.load(path_units)
            
            # 修复：如果第一个维度是1，去掉它
            units = units_data
            if units.shape[0] == 1:
                units = units[0]
            
            units = repeat_expand_2d(units, f0.size(0), self.unit_interpolate_mode).transpose(0, 1)
            
        units = units[start_frame : start_frame + units_frame_len]

        # load volume
        vol_key = 'aug_vol' if aug_flag else 'volume'
        volume = data_buffer.get(vol_key)
        volume_frames = volume[start_frame : start_frame + units_frame_len]
        

        # load domain_label
        domain_label = data_buffer.get('domain_label')
        
        # load f0 distribution labels
        f0_dist = data_buffer.get('f0_dist')
        
        # load speaker embedding
        spk_embd = data_buffer.get('spk_embd')
        
        # load shift
        aug_shift = torch.from_numpy(np.array([[aug_shift]])).float()
        
        result_dict = dict(
            mel=mel, 
            f0=f0_frames, 
            volume=volume_frames, 
            units=units, 
            aug_shift=aug_shift, 
            domain_label=domain_label, 
            f0_dist=f0_dist, 
            spk_embd=spk_embd,
            name=name, 
            name_ext=name_ext
        )
        
        return result_dict

    def __len__(self):
        return len(self.paths)