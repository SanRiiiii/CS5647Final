import time

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast
from torch.cuda.amp import GradScaler

import utils
from saver import Saver
from grl import GradientReversal, get_grl_lambda


def debug_nan_loss(data, model, loss, loss_f0_adv, grl_lambda, 
                   spk_embd, f0_dist_logits, step):
    """
    详细追踪NaN的来源
    """
    print("\n" + "="*80)
    print(f"🔴 NaN DETECTED at Step {step}")
    print("="*80)
    
    # 1. 检查输入数据
    print("\n[1] Input Data Check:")
    for key in ['units', 'f0', 'volume', 'mel']:
        if key in data:
            tensor = data[key]
            print(f"  {key:8s}: shape={tensor.shape}, "
                  f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, "
                  f"mean={tensor.mean().item():.6f}, "
                  f"has_nan={torch.isnan(tensor).any().item()}, "
                  f"has_inf={torch.isinf(tensor).any().item()}")
    
    # 2. 检查模型输出
    print("\n[2] Model Output Check:")
    if spk_embd is not None:
        print(f"  spk_embd     : shape={spk_embd.shape}, "
              f"min={spk_embd.min().item():.6f}, max={spk_embd.max().item():.6f}, "
              f"has_nan={torch.isnan(spk_embd).any().item()}")
    if f0_dist_logits is not None:
        print(f"  f0_dist_logits: shape={f0_dist_logits.shape}, "
              f"min={f0_dist_logits.min().item():.6f}, max={f0_dist_logits.max().item():.6f}, "
              f"has_nan={torch.isnan(f0_dist_logits).any().item()}")
    
    # 3. 检查损失
    print("\n[3] Loss Check:")
    print(f"  main loss    : {loss.item() if not torch.isnan(loss) else 'NaN'}")
    print(f"  f0_adv_loss  : {loss_f0_adv.item()}")
    print(f"  grl_lambda   : {grl_lambda}")
    
    # 4. 检查F0分布标签
    if 'f0_dist' in data and data['f0_dist'] is not None:
        print(f"\n[4] F0 Distribution Labels Check:")
        f0_dist = data['f0_dist']
        print(f"  f0_dist shape: {f0_dist.shape}")
        print(f"  f0_dist values (5 percentiles): {f0_dist[0].detach().cpu().numpy() if f0_dist.shape[0] > 0 else 'N/A'}")
        print(f"  Class distribution: {torch.bincount(f0_dist.flatten(), minlength=5).detach().cpu().numpy()}")
    
    print("\n" + "="*80)
    print("💡 Suggested Actions:")
    print("  1. Check if f0_adv_loss is abnormally high (sudden spike)")
    print("  2. Reduce grl_lambda growth speed (lower grl_gamma)")
    print("  3. Add stronger gradient clipping for spk_transformer")
    print("  4. Check F0 distribution labels in this batch")
    print("="*80 + "\n")


# get_grl_lambda函数已移至grl.py模块


def test(args, model, vocoder, loader_test, saver):
    print(' [*] testing...')
    model.eval()

    # losses
    test_loss = 0.
    
    # intialization
    num_batches = len(loader_test)
    rtf_all = []
    
    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            try:
                fn = data['name'][0].split("/")[-1]
                speaker = data['name'][0].split("/")[-2]
                print('--------')
                print('{}/{} - {}'.format(bidx, num_batches, fn))

                # unpack data
                for k in data.keys():
                    if not k.startswith('name'):
                        data[k] = data[k].to(args.device)
                print('>>', data['name'][0])

                # forward
                st_time = time.time()
            
                # 使用预加载的说话人嵌入
                spk_embd = data.get('spk_embd')
                mel, attention_gate = model(
                        data['units'], 
                        data['f0'], 
                        data['volume'], 
                        spk_embd=spk_embd,
                        aug_shift=None,  # 验证时不使用aug_shift
                        gt_spec=data['mel'],
                        infer=True
                        )
                print(f"[VAL DEBUG] Model returned, mel.shape = {mel.shape}")
                signal = vocoder.infer(mel, data['f0'])
                ed_time = time.time()
                            
                # RTF
                run_time = ed_time - st_time
                song_time = signal.shape[-1] / args.data.sampling_rate
                rtf = run_time / song_time
                print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
                rtf_all.append(rtf)
               
                # loss
                # 循环多次对扩散模型的随机loss进行蒙特卡洛采样
                for i in range(args.train.batch_size):
                    try:
                        loss, _ = model(
                            data['units'], 
                            data['f0'], 
                            data['volume'], 
                            spk_embd=spk_embd,
                            aug_shift=None,  # 验证时不使用aug_shift
                            gt_spec=data['mel'],
                            infer=False)
                        if isinstance(loss, list):
                            test_loss += loss[0].item()
                        else:
                            test_loss += loss.item()
        
                    except Exception as e:
                        print(f"[VAL DEBUG LOOP] ❌ Batch {bidx}, Loop {i}: FAILED with error: {e}")
                        # 跳过这次迭代，继续下一次
                        continue
                
                # log mel
                saver.log_spec(f"{speaker}_{fn}.wav", data['mel'], mel)
                
                # log audi
                try:
                    path_audio = data['name_ext'][0]
                    audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
                    if len(audio.shape) > 1:
                        audio = librosa.to_mono(audio)
                    audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
                    saver.log_audio({f"{speaker}_{fn}_gt.wav": audio,f"{speaker}_{fn}_pred.wav": signal})
                except Exception as e:
                    print(f"Warning: Failed to load audio for logging: {e}")
                    print(f"  - Path: {path_audio}")
                    # 继续验证，只是跳过音频日志
                
            except Exception as e:
                print(f"Warning: Validation failed for batch {bidx}: {e}")
                print(f"  - File: {data['name'][0] if 'name' in data else 'Unknown'}")
                continue
                
    # report
    # 扩散模型的loss是随机的，循环多次采样后求平均
    test_loss /= args.train.batch_size
    test_loss /= num_batches 
    
    # check
    print(' [test_loss] test_loss:', test_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    return test_loss


# def train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_test):
def train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_test):

    # 训练稳定性监控（移除早停机制以提高训练速度）
    loss_history = []

    # saver
    save_dir = args.env.expdir
    saver = Saver(args, save_dir, initial_global_step=initial_global_step)

    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    
    # run
    num_batches = len(loader_train)
    model.train()
    saver.log_info('======= start training =======')
    scaler = GradScaler()
    if args.train.amp_dtype == 'fp32': # fp32
        dtype = torch.float32
    elif args.train.amp_dtype == 'fp16':
        dtype = torch.float16
    elif args.train.amp_dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(' [x] Unknown amp_dtype: ' + args.train.amp_dtype)
    saver.log_info("epoch|batch_idx/num_batches|output_dir|batch/s|lr|time|step")
    saver.log_info(f"Training will run for {args.train.epochs} epochs")
    
    for epoch in range(args.train.epochs):
        saver.log_info(f"\n{'='*80}")
        saver.log_info(f"Starting Epoch {epoch + 1}/{args.train.epochs}")
        saver.log_info(f"{'='*80}\n")
        
        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()

            # unpack data
            for k in data.keys():
                if not k.startswith('name'):
                    data[k] = data[k].to(args.device)
            
            # 添加数值稳定性检查
            if torch.isnan(data['units']).any() or torch.isinf(data['units']).any():
                print(f"Warning: Invalid units detected, skipping batch")
                continue
            if torch.isnan(data['f0']).any() or torch.isinf(data['f0']).any():
                print(f"Warning: Invalid f0 detected, skipping batch")
                continue
            if torch.isnan(data['volume']).any() or torch.isinf(data['volume']).any():
                print(f"Warning: Invalid volume detected, skipping batch")
                continue
            if torch.isnan(data['mel']).any() or torch.isinf(data['mel']).any():
                print(f"Warning: Invalid mel detected, skipping batch")
                continue
            
            # forward
            # 使用预加载的说话人嵌入
            spk_embd = data.get('spk_embd')  # [B, n_hidden]
            
            if dtype == torch.float32:
                loss, attention_gate = model(data['units'].float(), data['f0'], data['volume'], 
                                spk_embd=spk_embd.float() if spk_embd is not None else None,
                                aug_shift = data['aug_shift'], 
                                gt_spec=data['mel'].float(), infer=False)
            else:
                with autocast(device_type=args.device, dtype=dtype):
                    loss, attention_gate = model(data['units'], data['f0'], data['volume'], 
                                    spk_embd=spk_embd, aug_shift = data['aug_shift'], 
                                    gt_spec=data['mel'], infer=False)
            
            # 检查主模型损失
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Main model loss is NaN/Inf: {loss.item()}")
                print(f"  - Main model loss type: {type(loss)}")
                if hasattr(loss, 'shape'):
                    print(f"  - Main model loss shape: {loss.shape}")
                
                # 检查F0 log变换
                f0_test = data['f0']
                f0_log_test = (1 + f0_test / 700).log()
                if torch.isnan(f0_log_test).any():
                    print(f"  - F0 log transform produces NaN!")
                    print(f"    - F0 min: {f0_test.min().item():.6f}, max: {f0_test.max().item():.6f}")
                    print(f"    - F0 log min: {f0_log_test.min().item():.6f}, max: {f0_log_test.max().item():.6f}")
                
                continue
            # --- Auxiliary losses: Domain Adversarial + F0 stats ---
            
            # F0对抗训练参数（独立控制）
            f0_adv_cfg = getattr(args.train, 'f0_adversarial_training', {})
            f0_adv_enabled = f0_adv_cfg.get('enabled', False)
            f0_adv_start_step = f0_adv_cfg.get('start_step', 20000)
            f0_adv_total_steps = f0_adv_cfg.get('total_steps', 100000)
            f0_adv_loss_weight = f0_adv_cfg.get('loss_weight', 0.05)
            
            # GRL参数（独立控制）
            grl_cfg = getattr(args.train, 'grl_scheduling', {})
            grl_gamma = grl_cfg.get('gamma', 10.0)
            grl_max_lambda = grl_cfg.get('max_lambda', 0.12)
            
            # 计算当前的GRL lambda（使用DANN的平滑调度）
            grl_lambda = get_grl_lambda(
                current_step=saver.global_step,
                start_step=f0_adv_start_step,
                total_steps=f0_adv_total_steps,
                gamma=grl_gamma,
                max_lambda=grl_max_lambda
            )
            
            # 判断是否使用F0对抗训练
            should_use_f0_adv = f0_adv_enabled and grl_lambda > 0
            

            # 初始化F0对抗损失
            loss_f0_adv = torch.tensor(0.0, device=loss.device)
            f0_acc = 0.0  # F0分类准确率

            # 处理F0对抗训练（使用spk_embd_transformer）
            if should_use_f0_adv:
                try:
                    # 使用F0分布分类（与spk_encoder的F0分布分类一致）
                    f0_dist_target = data['f0_dist']  # [B, 5] - 预计算的F0分布标签
                    if f0_dist_target is not None and hasattr(model, 'spk_transformer') and model.spk_transformer is not None and spk_embd is not None:
                        # 通过GRL的F0分布预测器预测
                        _, f0_dist_pred_grl = model.spk_transformer(spk_embd, grl_lambda=grl_lambda)
                        
                        if dtype != torch.float32:
                            f0_dist_pred_grl = f0_dist_pred_grl.float()
                        
                        # 计算交叉熵损失（对5个分位数分别计算，然后取平均）
                        f0_dist_logits_reshaped = f0_dist_pred_grl.reshape(-1, 5)  # [B*5, 5]
                        f0_dist_target_reshaped = f0_dist_target.reshape(-1)  # [B*5]
                        
                        # 裁剪logits到[-5, 5]范围，增加数值稳定性
                        f0_dist_logits_reshaped = torch.clamp(f0_dist_logits_reshaped, -5, 5)
                        
                        # 计算交叉熵损失（带标签平滑0.1）
                        loss_f0_adv = F.cross_entropy(f0_dist_logits_reshaped, f0_dist_target_reshaped, label_smoothing=0.1)
                        
                        # 计算F0分类准确率
                        f0_pred_classes = torch.argmax(f0_dist_logits_reshaped, dim=1)
                        f0_acc = (f0_pred_classes == f0_dist_target_reshaped).float().mean().item()
                    else:
                        loss_f0_adv = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
                        f0_acc = 0.0
                except Exception as e:
                    print(f"Warning: F0 adversarial training failed: {e}")
                    loss_f0_adv = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
                    f0_acc = 0.0
            else:
                loss_f0_adv = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
                f0_acc = 0.0

            # ✅ 组合损失：重建损失 + F0对抗损失
            loss = loss + f0_adv_loss_weight * loss_f0_adv
            #     loss_mel = loss[0]*50 
            #     loss_pitch = loss[1] 
            #     loss = loss_mel +loss_pitch
            # loss=loss*1000
            # ✅ 检测NaN loss并停止训练
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n🔴 NaN/Inf detected in main loss at step {saver.global_step}")
                print(f"   Loss value: {loss.item() if not torch.isnan(loss) else 'NaN'}")
                print("\n❌ Training stopped immediately due to NaN loss")
                raise Exception(f"NaN loss detected at step {saver.global_step} - Training stopped")
            
            # backpropagate with gradient clipping
            # ✅ 初始化梯度范数变量（会在backward后计算）
            f0_grad_norm = 0.0
            
            if dtype == torch.float32:
                # 正常backward
                loss.backward()
                
                # ✅ 裁剪spk_transformer的梯度（如果存在）
                if hasattr(model, 'spk_transformer') and model.spk_transformer is not None:
                    torch.nn.utils.clip_grad_norm_(model.spk_transformer.parameters(), max_norm=1.0)
                
                # ✅ 在梯度裁剪之后计算spk_transformer梯度范数
                if saver.global_step % args.train.interval_log == 0 and hasattr(model, 'spk_transformer') and model.spk_transformer is not None:
                    for p in model.spk_transformer.parameters():
                        if p.grad is not None:
                            f0_grad_norm += p.grad.data.norm(2).item() ** 2
                    f0_grad_norm = f0_grad_norm ** 0.5
                
                optimizer.step()
            else:
                # 正常backward
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                # ✅ 裁剪spk_transformer的梯度（如果存在）
                if hasattr(model, 'spk_transformer') and model.spk_transformer is not None:
                    torch.nn.utils.clip_grad_norm_(model.spk_transformer.parameters(), max_norm=1.0)
                
                # ✅ 在梯度裁剪之后计算spk_transformer梯度范数
                if saver.global_step % args.train.interval_log == 0 and hasattr(model, 'spk_transformer') and model.spk_transformer is not None:
                    for p in model.spk_transformer.parameters():
                        if p.grad is not None:
                            f0_grad_norm += p.grad.data.norm(2).item() ** 2
                    f0_grad_norm = f0_grad_norm ** 0.5
                
                scaler.step(optimizer)
                scaler.update()
            
            scheduler.step()
                
            # log loss
            if saver.global_step % args.train.interval_log == 0:
                current_lr =  optimizer.param_groups[0]['lr']
                
                # 构建详细的日志信息
                training_phase = "PRETRAIN" if grl_lambda == 0.0 else ("WARMUP" if grl_lambda < 0.99 else "FULL_ADV")
                
                log_info = (
                    'epoch: {} | {:3d}/{:3d} | {} | batch/s: {:.2f} | lr: {:.6} | '
                    'loss: {:.3f} | f0_adv_loss: {:.3f} | '
                    'f0_acc: {:.3f} | grl_λ: {:.4f} | '
                    'f0_grad: {:.3f} | gate: {:.4f} | '
                    'phase: {} | time: {} | step: {}'
                ).format(
                    epoch,
                    batch_idx,
                    num_batches,
                    save_dir,
                    args.train.interval_log/saver.get_interval_time(),
                    current_lr,
                    loss.item(),
                    loss_f0_adv.item(),
                    f0_acc,
                    grl_lambda,
                    f0_grad_norm,
                    attention_gate.item(),
                    training_phase,
                    saver.get_total_time(),
                    saver.global_step
                )
                
                saver.log_info(log_info)
                
                saver.log_value({
                    'train/loss': loss.item(),
                    'train/f0_adv_loss': loss_f0_adv.item(),
                    'train/f0_acc': f0_acc,
                    'train/grl_lambda': grl_lambda,
                    'train/f0_grad_norm': f0_grad_norm,
                    'train/attention_gate': attention_gate.item(),
                })

                
                saver.log_value({
                    'train/lr': current_lr
                })
            
            # model saving (skip validation for faster training)
            if saver.global_step % args.train.interval_val == 0:
                optimizer_save = optimizer if args.train.save_opt else None
                
                # save latest model
                saver.save_model(model, optimizer_save, postfix=f'{saver.global_step}')
                last_val_step = saver.global_step - args.train.interval_val
                if last_val_step % args.train.interval_force_save != 0:
                    saver.delete_model(postfix=f'{last_val_step}')
                
                # Skip validation for faster training
                saver.log_info(f"Model saved at step {saver.global_step} (validation skipped)")
                
                model.train()
    
    # 训练完成
    saver.log_info("\n" + "="*80)
    saver.log_info("🎉 Training completed successfully!")
    saver.log_info(f"✅ Finished {args.train.epochs} epochs")
    saver.log_info(f"✅ Total steps: {saver.global_step}")
    saver.log_info(f"✅ Total time: {saver.get_total_time()}")
    saver.log_info("="*80 + "\n")
    
    # 保存最终模型
    optimizer_save = optimizer if args.train.save_opt else None
    saver.save_model(model, optimizer_save, postfix='final')
    saver.log_info(f"💾 Final model saved as: model_final.pt")
    
    saver.log_info("\n🚀 Training script will now exit with code 0")
                          
