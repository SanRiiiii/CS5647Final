#!/usr/bin/env python3

import argparse
import logging
import soundfile
import os

import infer_tool
from infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description='ComoSVC Zero-Shot Inference')
    parser.add_argument('-ts', '--total_steps', type=int, default=1, 
                       help='the total number of iterative steps during inference')
    parser.add_argument('--clip', type=float, default=0, 
                       help='Slicing the audios which are to be converted')
    parser.add_argument('-k', '--key_shift', type=int, default=0, 
                       help='Key shift for pitch adjustment')
    parser.add_argument('--source_audio', type=str, required=True,
                       help='Path to source audio file')
    parser.add_argument('--target_speaker', type=str, required=True,
                       help='Path to target speaker reference audio')
    parser.add_argument('-m', '--model_path', type=str, default="./logs/model_800000.pt", 
                       help='the path to checkpoint of ComoSVC')
    parser.add_argument('-c', '--config_path', type=str, default="./logs/config.yaml", 
                       help='the path to config file of ComoSVC')

    args = parser.parse_args()

    slice_db = -40 
    wav_format = 'wav'
    pad_seconds = 0.5

    print(f"Loading ComoSVC model from {args.model_path}")
    svc_model = Svc(args.model_path, args.config_path, args.total_steps)
    
    resultfolder = 'result'
    infer_tool.mkdir([resultfolder])
    
    print("Zero-shot voice conversion mode")
    print(f"Source audio: {args.source_audio}")
    print(f"Target speaker: {args.target_speaker}")
    
    try:
        audio = svc_model.zero_shot_inference(
            source_audio_path=args.source_audio,
            target_speaker_path=args.target_speaker,
            key_shift=args.key_shift,
            slice_db=slice_db,
            pad_seconds=pad_seconds,
            clip_seconds=args.clip
        )
        
        step_num = args.model_path.split('/')[-1].split('.')[0]
        source_name = os.path.splitext(os.path.basename(args.source_audio))[0]
        target_name = os.path.splitext(os.path.basename(args.target_speaker))[0]
        res_path = f'{resultfolder}/{source_name}_to_{target_name}_{step_num}.{wav_format}'
        soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
        print(f"Zero-shot conversion completed! Saved to: {res_path}")
        
        svc_model.clear_empty()
        
    except Exception as e:
        print(f"Error during zero-shot conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
