import os
import shutil
from pathlib import Path

DATASET_ROOT = {

    "m4singer": {
        "src_root": Path("./m4singer"),
        "dst_root": Path("./dataset_raw")
    },
    'vctk': {
        "src_root": Path("./VCTK-Corpus/VCTK-Corpus/wav48"),
        "dst_root": Path("./dataset_raw")
    }
}

# 原始m4singer数据集路径
for dataset in DATASET_ROOT:
    src_root = DATASET_ROOT[dataset]["src_root"]
    dst_root = DATASET_ROOT[dataset]["dst_root"]
    dst_root.mkdir(parents=True, exist_ok=True)
    if dataset == "m4singer":

        singer_ids = {}

        # 遍历所有歌手
        for singer in src_root.iterdir():
                if not singer.is_dir():
                    continue
                singer_id = singer.name.split('#')[0]
                if singer_id in singer_ids:
                    singer_folder = singer_ids[singer_id]
                else:
                    singer_folder = dst_root / singer_id
                    singer_ids[singer_id]= singer_folder
                    singer_folder.mkdir(parents=True, exist_ok=True)

                # 遍历 read/wav 和 sing/wav
            
                for wav_file in singer.glob("*.wav"):
                        shutil.copy(wav_file, singer_folder / wav_file.name)

    elif dataset == "vctk":

        for speaker in src_root.iterdir():
            if not speaker.is_dir():
                continue
            speaker_id = speaker.name
            speaker_folder = dst_root / speaker_id
            speaker_folder.mkdir(parents=True, exist_ok=True)
            for wav_file in speaker.glob("*.wav"):
                shutil.copy(wav_file, speaker_folder / wav_file.name)

