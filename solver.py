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


# get_grl_lambda function has been moved to grl.py module


def test(args, model, vocoder, loader_test, saver):
    saver.log_info(' [*] testing...')
    model.eval()

    # losses
    test_loss = 0.
    
    # initialization
    num_batches = len(loader_test)
    rtf_all = []
    
    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            try:
                fn = data['name'][0].split("/")[-1]
                speaker = data['name'][0].split("/")[-2]

                # unpack data
                for k in data.keys():
                    if not k.startswith('name'):
                        data[k] = data[k].to(args.device)

                # forward
                st_time = time.time()
            
                # Use preloaded speaker embedding
                spk_embd = data.get('spk_embd')
                mel, attention_gate = model(
                        data['units'], 
                        data['f0'], 
                        data['volume'], 
                        spk_embd=spk_embd,
                        aug_shift=None,  # No aug_shift during validation
                        gt_spec=data['mel'],
                        infer=True
                        )
                signal = vocoder.infer(mel, data['f0'])
                ed_time = time.time()
                            
                # RTF
                run_time = ed_time - st_time
                song_time = signal.shape[-1] / args.data.sampling_rate
                rtf = run_time / song_time
                rtf_all.append(rtf)
               
                # loss
                # Monte Carlo sampling for diffusion model's stochastic loss
                for i in range(args.train.batch_size):
                    try:
                        loss, _ = model(
                            data['units'], 
                            data['f0'], 
                            data['volume'], 
                            spk_embd=spk_embd,
                            aug_shift=None,  # No aug_shift during validation
                            gt_spec=data['mel'],
                            infer=False)
                        if isinstance(loss, list):
                            test_loss += loss[0].item()
                        else:
                            test_loss += loss.item()
        
                    except Exception as e:
                        # Skip this iteration and continue to next
                        raise e
                
                # log mel
                saver.log_spec(f"{speaker}_{fn}.wav", data['mel'], mel)
                
                # log audio
                try:
                    path_audio = data['name_ext'][0]
                    audio, sr = librosa.load(path_audio, sr=args.data.sampling_rate)
                    if len(audio.shape) > 1:
                        audio = librosa.to_mono(audio)
                    audio = torch.from_numpy(audio).unsqueeze(0).to(signal)
                    saver.log_audio({f"{speaker}_{fn}_gt.wav": audio,f"{speaker}_{fn}_pred.wav": signal})
                except Exception as e:
                    raise e
                
            except Exception as e:
                raise e
                
    # report
    # Average loss from multiple Monte Carlo samples
    test_loss /= args.train.batch_size
    test_loss /= num_batches 
    
    # check
    saver.log_info(f' [test_loss] test_loss: {test_loss}')
    saver.log_info(f' Real Time Factor: {np.mean(rtf_all)}')
    return test_loss


# def train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_test):
def train(args, initial_global_step, model, optimizer, scheduler, vocoder, loader_train, loader_test):

    # Training stability monitoring (early stopping removed for faster training)
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
            
            # Add numerical stability check
            if torch.isnan(data['units']).any() or torch.isinf(data['units']).any():
                print("NaN/Inf detected in units")
            if torch.isnan(data['f0']).any() or torch.isinf(data['f0']).any():
                print("NaN/Inf detected in f0")
            if torch.isnan(data['volume']).any() or torch.isinf(data['volume']).any():
                print("NaN/Inf detected in volume")
            if torch.isnan(data['mel']).any() or torch.isinf(data['mel']).any():
                print("NaN/Inf detected in mel")
            
            # forward
            # Use preloaded speaker embedding
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
            
            # Check main model loss
            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN/Inf detected in loss")
            # --- Auxiliary losses: Domain Adversarial + F0 stats ---
            
            # F0 adversarial training parameters (independent control)
            f0_adv_cfg = getattr(args.train, 'f0_adversarial_training', {})
            f0_adv_enabled = f0_adv_cfg.get('enabled', False)
            f0_adv_start_step = f0_adv_cfg.get('start_step', 20000)
            f0_adv_total_steps = f0_adv_cfg.get('total_steps', 100000)
            f0_adv_loss_weight = f0_adv_cfg.get('loss_weight', 0.05)
            
            # GRL parameters (independent control)
            grl_cfg = getattr(args.train, 'grl_scheduling', {})
            grl_gamma = grl_cfg.get('gamma', 10.0)
            grl_max_lambda = grl_cfg.get('max_lambda', 0.12)
            
            # Compute current GRL lambda (using DANN's smooth scheduling)
            grl_lambda = get_grl_lambda(
                current_step=saver.global_step,
                start_step=f0_adv_start_step,
                total_steps=f0_adv_total_steps,
                gamma=grl_gamma,
                max_lambda=grl_max_lambda
            )
            
            # Determine whether to use F0 adversarial training
            should_use_f0_adv = f0_adv_enabled and grl_lambda > 0
            

            # Initialize F0 adversarial loss
            loss_f0_adv = torch.tensor(0.0, device=loss.device)
            f0_acc = 0.0  # F0 classification accuracy

            # Process F0 adversarial training (using spk_embd_transformer)
            if should_use_f0_adv:
                try:
                    # Use F0 distribution classification (consistent with spk_encoder's F0 distribution classification)
                    f0_dist_target = data['f0_dist']  # [B, 5] - Pre-computed F0 distribution labels
                    if f0_dist_target is not None and hasattr(model, 'spk_transformer') and model.spk_transformer is not None and spk_embd is not None:
                        # Predict through GRL's F0 distribution predictor
                        _, f0_dist_pred_grl = model.spk_transformer(spk_embd, grl_lambda=grl_lambda)
                        
                        if dtype != torch.float32:
                            f0_dist_pred_grl = f0_dist_pred_grl.float()
                        
                        # Compute cross-entropy loss (separately for 5 quantiles, then average)
                        f0_dist_logits_reshaped = f0_dist_pred_grl.reshape(-1, 5)  # [B*5, 5]
                        f0_dist_target_reshaped = f0_dist_target.reshape(-1)  # [B*5]
                        
                        # Clip logits to [-5, 5] range for numerical stability
                        f0_dist_logits_reshaped = torch.clamp(f0_dist_logits_reshaped, -5, 5)
                        
                        # Compute cross-entropy loss (with label smoothing 0.1)
                        loss_f0_adv = F.cross_entropy(f0_dist_logits_reshaped, f0_dist_target_reshaped, label_smoothing=0.1)
                        
                        # Compute F0 classification accuracy
                        f0_pred_classes = torch.argmax(f0_dist_logits_reshaped, dim=1)
                        f0_acc = (f0_pred_classes == f0_dist_target_reshaped).float().mean().item()
                    else:
                        loss_f0_adv = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
                        f0_acc = 0.0
                except Exception as e:
                    loss_f0_adv = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
                    f0_acc = 0.0
            else:
                loss_f0_adv = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
                f0_acc = 0.0

            # Combine loss: reconstruction loss + F0 adversarial loss
            loss = loss + f0_adv_loss_weight * loss_f0_adv
            #     loss_mel = loss[0]*50 
            #     loss_pitch = loss[1] 
            #     loss = loss_mel +loss_pitch
            # loss=loss*1000
            # Detect NaN loss and stop training
            if torch.isnan(loss) or torch.isinf(loss):
                saver.log_info(f"\nNaN/Inf detected in main loss at step {saver.global_step}")
                saver.log_info(f"Loss value: {loss.item() if not torch.isnan(loss) else 'NaN'}")
                saver.log_info("\nTraining stopped immediately due to NaN loss")
                raise Exception(f"NaN loss detected at step {saver.global_step} - Training stopped")
            
            # backpropagate with gradient clipping
            # Initialize gradient norm variable (computed after backward)
            f0_grad_norm = 0.0
            
            if dtype == torch.float32:
                # Normal backward
                loss.backward()
                
                # Clip spk_transformer gradients (if exists)
                if hasattr(model, 'spk_transformer') and model.spk_transformer is not None:
                    torch.nn.utils.clip_grad_norm_(model.spk_transformer.parameters(), max_norm=1.0)
                
                # Compute spk_transformer gradient norm after clipping
                if saver.global_step % args.train.interval_log == 0 and hasattr(model, 'spk_transformer') and model.spk_transformer is not None:
                    for p in model.spk_transformer.parameters():
                        if p.grad is not None:
                            f0_grad_norm += p.grad.data.norm(2).item() ** 2
                    f0_grad_norm = f0_grad_norm ** 0.5
                
                optimizer.step()
            else:
                # Normal backward
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                # Clip spk_transformer gradients (if exists)
                if hasattr(model, 'spk_transformer') and model.spk_transformer is not None:
                    torch.nn.utils.clip_grad_norm_(model.spk_transformer.parameters(), max_norm=1.0)
                
                # Compute spk_transformer gradient norm after clipping
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
                
                # Build detailed log info
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
    
    # Training completed
    saver.log_info("\n" + "="*80)
    saver.log_info("Training completed successfully!")
    saver.log_info(f"Finished {args.train.epochs} epochs")
    saver.log_info(f"Total steps: {saver.global_step}")
    saver.log_info(f"Total time: {saver.get_total_time()}")
    saver.log_info("="*80 + "\n")
    
    # Save final model
    optimizer_save = optimizer if args.train.save_opt else None
    saver.save_model(model, optimizer_save, postfix='final')
    saver.log_info(f"Final model saved as: model_final.pt")
    
    saver.log_info("\nTraining script will now exit with code 0")
                          
