#!/usr/bin/env python3
"""
简化的ComoSVC推理脚本
移除了teacher/student模型区分，只使用统一的ComoSVC模型
"""

import argparse
import logging
import soundfile
import os

import infer_tool
from infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description='ComoSVC inference')
    parser.add_argument('-ts', '--total_steps', type=int, default=1, 
                       help='the total number of iterative steps during inference')
    parser.add_argument('--clip', type=float, default=0, 
                       help='Slicing the audios which are to be converted')
    parser.add_argument('-n', '--clean_names', type=str, nargs='+', default=['1.wav'], 
                       help='The audios to be converted, should be put in "raw" directory')
    parser.add_argument('-k', '--keys', type=int, nargs='+', default=[0], 
                       help='To Adjust the Key')
    parser.add_argument('-s', '--spk_list', type=str, nargs='+', default=['singer1'], 
                       help='The target singer')
    
    # Zero-shot语音转换参数
    parser.add_argument('--zero_shot', action='store_true',
                       help='Enable zero-shot voice conversion mode')
    parser.add_argument('--source_audio', type=str, default=None,
                       help='Path to source audio file (for zero-shot mode)')
    parser.add_argument('--target_speaker', type=str, default=None,
                       help='Path to target speaker reference audio (for zero-shot mode)')
    parser.add_argument('-m', '--model_path', type=str, default="./logs/model_800000.pt", 
                       help='the path to checkpoint of ComoSVC')
    parser.add_argument('-c', '--config_path', type=str, default="./logs/config.yaml", 
                       help='the path to config file of ComoSVC')

    args = parser.parse_args()

    # 设置参数
    clean_names = args.clean_names
    keys = args.keys
    spk_list = args.spk_list
    slice_db = -40 
    wav_format = 'wav'
    pad_seconds = 0.5
    clip = args.clip

    # 初始化模型
    print(f"Loading ComoSVC model from {args.model_path}")
    svc_model = Svc(args.model_path, args.config_path, args.total_steps)
    
    # 创建输出目录
    resultfolder = 'result'
    infer_tool.mkdir(["raw", resultfolder])
    
    # Zero-shot语音转换模式
    if args.zero_shot:
        if not args.source_audio or not args.target_speaker:
            print("Error: Zero-shot mode requires both --source_audio and --target_speaker")
            return
        
        print("🎵 Zero-shot voice conversion mode")
        print(f"📁 Source audio: {args.source_audio}")
        print(f"🎤 Target speaker: {args.target_speaker}")
        
        try:
            # 执行zero-shot推理
            audio = svc_model.zero_shot_inference(
                source_audio_path=args.source_audio,
                target_speaker_path=args.target_speaker,
                key_shift=keys[0] if keys else 0,
                slice_db=slice_db,
                pad_seconds=pad_seconds,
                clip_seconds=clip
            )
            
            # 保存结果
            step_num = args.model_path.split('/')[-1].split('.')[0]
            source_name = os.path.splitext(os.path.basename(args.source_audio))[0]
            target_name = os.path.splitext(os.path.basename(args.target_speaker))[0]
            res_path = f'{resultfolder}/{source_name}_to_{target_name}_{step_num}.{wav_format}'
            soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
            print(f"✅ Zero-shot conversion completed! Saved to: {res_path}")
            
            # 清理内存
            svc_model.clear_empty()
            
        except Exception as e:
            print(f"❌ Error during zero-shot conversion: {e}")
            import traceback
            traceback.print_exc()
    
    # 传统模式（保持原有逻辑）
    else:
        # 处理音频
        infer_tool.fill_a_to_b(keys, clean_names)
        
        for clean_name, tran in zip(clean_names, keys):
            raw_audio_path = f"raw/{clean_name}"
            if "." not in raw_audio_path:
                raw_audio_path += ".wav"
            
            # 格式化音频
            infer_tool.format_wav(raw_audio_path)
            
            for spk in spk_list:
                print(f"Converting {clean_name} to speaker {spk} with key shift {tran}")
                
                kwarg = {
                    "raw_audio_path": raw_audio_path,
                    "spk": spk,
                    "tran": tran,
                    "slice_db": slice_db,
                    "pad_seconds": pad_seconds,
                    "clip_seconds": clip,
                }
                
                # 执行推理
                audio = svc_model.slice_inference(**kwarg)
                
                # 保存结果
                step_num = args.model_path.split('/')[-1].split('.')[0]
                res_path = f'{resultfolder}/{clean_name}_{spk}_{step_num}.{wav_format}'
                soundfile.write(res_path, audio, svc_model.target_sample, format=wav_format)
                print(f"Saved result to {res_path}")
                
                # 清理内存
                svc_model.clear_empty()


if __name__ == '__main__':
    main()
