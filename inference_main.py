import logging
import soundfile
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'

import infer_tool
from infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='CoMoSVC Inference - Voice Conversion using Pre-generated Speaker Embeddings')
    parser.add_argument('-ts', '--total_steps', type=int, default=1, help='the total number of iterative steps during inference')
    parser.add_argument('--clip', type=float, default=0, help='Slicing the audios which are to be converted')
    parser.add_argument('-n','--clean_names', type=str, nargs='+', default=['test.wav'], help='Source audio file paths (relative or absolute)')
    parser.add_argument('-k','--keys', type=int, nargs='+', default=[0], help='To Adjust the Key (in semitones)')
    parser.add_argument('-s','--spk_list', type=str, nargs='+', default=['p225'], help='The target speaker names (must match folder names in dataset)')
    parser.add_argument('-m','--model_path', type=str, default="./logs/model_96000.pt", help='the path to checkpoint of ComoSVC')
    parser.add_argument('-c','--config_path', type=str, default="./configs/diffusion.yaml", help='the path to config file of ComoSVC')
    parser.add_argument('-d','--dataset_path', type=str, default="./dataset", help='the path to dataset directory (for loading speaker embeddings)')

    args = parser.parse_args()

    clean_names = args.clean_names
    keys = args.keys
    spk_list = args.spk_list
    slice_db = -40 
    wav_format = 'wav' # the format of the output audio
    pad_seconds = 0.5
    clip = args.clip

    model_path = args.model_path
    config_path = args.config_path
    dataset_path = args.dataset_path
    resultfolder = 'result'

    print("="*80)
    print("🎵 CoMoSVC Inference")
    print("="*80)
    print(f"📁 Model: {model_path}")
    print(f"⚙️  Config: {config_path}")
    print(f"📂 Dataset: {dataset_path}")
    print(f"🎤 Target speakers: {spk_list}")
    print(f"🎼 Key shifts: {keys}")
    print("="*80)

    svc_model = Svc(model_path, config_path, args.total_steps, dataset_path=dataset_path)
    
    infer_tool.mkdir(["raw", resultfolder])
    
    infer_tool.fill_a_to_b(keys, clean_names)
    for clean_name, tran in zip(clean_names, keys):
        # 直接使用传入的路径
        raw_audio_path = clean_name
        if "." not in raw_audio_path:
            raw_audio_path += ".wav"
        infer_tool.format_wav(raw_audio_path)
        
        for spk in spk_list:
            print(f"\n{'='*80}")
            print(f"Converting: {clean_name}")
            print(f"Target speaker: {spk}")
            print(f"Key shift: {tran:+d} semitones")
            print(f"{'='*80}\n")
            
            kwarg = {
                "raw_audio_path" : raw_audio_path,
                "spk" : spk,
                "tran" : tran,
                "slice_db" : slice_db,  # -40
                "pad_seconds" : pad_seconds,  # 0.5
                "clip_seconds" : clip,  # 0
            }
            
            audio = svc_model.slice_inference(**kwarg)
            
            # 生成输出文件名（只使用basename，不包含路径）
            step_num = model_path.split('/')[-1].split('.')[0]
            base_name = os.path.basename(clean_name).replace('.wav', '')
            output_name = f"{base_name}_{spk}_{step_num}"
            if tran != 0:
                output_name += f"_key{tran:+d}"
            res_path = f'{resultfolder}/{output_name}.{wav_format}'
            
            soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
            print(f"✅ Saved to: {res_path}\n")
            
            svc_model.clear_empty()
    
    print("="*80)
    print("🎉 All conversions completed!")
    print("="*80)
            
if __name__ == '__main__':
    main()
