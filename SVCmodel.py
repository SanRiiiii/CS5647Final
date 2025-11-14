import os
import torch
import torch.nn as nn
import yaml
from Vocoder import Vocoder
from como import Como
from mm_attention_fusion import MultiModalCrossAttention
from spk_embd_transformer import SpeakerEmbeddingTransformerWithGRL


class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__

    
def load_model_vocoder(
        model_path,
        device='cpu',
        config_path = None,
        total_steps=1
        ):
    if config_path is None:
        config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    else:
        config_file = config_path

    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    # load vocoder
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
    
    # load model
    use_attention = getattr(args.model, 'use_attention', True)
    print(f"Model Architecture Settings:")
    print(f"   use_attention: {use_attention}")
    print(f"   n_layers: {args.model.n_layers}")
    print(f"   n_chans: {args.model.n_chans}")
    print(f"   n_hidden: {args.model.n_hidden}")
    
    model = ComoSVC(
                args.data.encoder_out_channels, 
                args.model.use_pitch_aug,
                vocoder.dimension,
                args.model.n_layers,
                args.model.n_chans,
                args.model.n_hidden,
                total_steps,
                attention=use_attention,
                config=args
                )
    
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    
    # Check checkpoint parameters
    print("Checking checkpoint parameters...")
    ckpt_keys = set(ckpt['model'].keys())
    model_keys = set(model.state_dict().keys())
    
    print(f"Checkpoint has {len(ckpt_keys)} parameters")
    print(f"Model expects {len(model_keys)} parameters")
    
    # Check attention mechanism parameters
    attention_keys = [k for k in ckpt_keys if 'mm_attention' in k or 'gate_alpha_raw' in k]
    if attention_keys:
        print(f"Found attention parameters in checkpoint: {attention_keys}")
        for key in attention_keys:
            if 'gate_alpha_raw' in key:
                print(f"   {key}: {ckpt['model'][key].item():.6f}")
    else:
        print("No attention parameters found in checkpoint")
    
    # Load model parameters
    missing_keys, unexpected_keys = model.load_state_dict(ckpt['model'], strict=False)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    # Verify gate_alpha_raw loading
    if hasattr(model, 'mm_attention') and model.mm_attention is not None:
        gate_value = torch.sigmoid(model.mm_attention.gate_alpha_raw).item()
        print(f"Loaded gate_alpha_raw value: {gate_value:.6f}")
    
    model.eval()
    return model, vocoder, args


class SVCmodel(nn.Module):
    def __init__(
            self,
            input_channel,
            use_pitch_aug=True,
            out_dims=128, # define in como
            n_layers=20, 
            n_chans=384, 
            n_hidden=100,
            total_steps=1,
            attention=False,
            config=None
            ):
        super().__init__()

        self.unit_embed = nn.Linear(input_channel, n_hidden)
        
        self.f0_embed = nn.Linear(1, n_hidden)
        self.volume_embed = nn.Linear(1, n_hidden)
        self.f0_volume_embed = nn.Linear(2, n_hidden)

        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        else:
            self.aug_shift_embed = None
        
        self.spk_encoder = None
        
        self.n_hidden = n_hidden
        self.decoder = Como(out_dims, n_layers, n_chans, n_hidden, total_steps) 
        self.input_channel = input_channel

        if attention:
            if config and hasattr(config, 'model') and hasattr(config.model, 'attention'):
                attn_config = config.model.attention
                self.mm_attention = MultiModalCrossAttention(
                    d_model=n_hidden,
                    num_heads=attn_config.get('num_heads', 8),
                    dropout=attn_config.get('dropout', 0.1),
                    init_alpha=attn_config.get('init_alpha', 1.0)
                )
            else:
                self.mm_attention = MultiModalCrossAttention(
                    d_model=n_hidden,
                    num_heads=8,
                    dropout=0.1,
                    init_alpha=1.0
                )
        else:
            self.mm_attention = None

        self.spk_transformer = None
        self.spk_transformer_weights = None
        if config and hasattr(config, 'model') and hasattr(config.model, 'spk_embd_transformer'):
            tcfg = config.model.spk_embd_transformer
            if tcfg.get('enabled', False):
                self.spk_transformer = SpeakerEmbeddingTransformerWithGRL(
                    spk_embd_dim=n_hidden,
                    output_dim=n_hidden,
                    f0_pred_dim=1,
                    transform_type=tcfg.get('transform_type', 'linear'),
                    transform_config=tcfg.get('transform_config', {}),
                    dropout_rate=0.1,
                )
                self.spk_transformer_weights = tcfg.get('combine_weights', [1, 2])
        

    def forward(self, units, f0, volume, ref_mel=None, speaker_id=None, aug_shift=None,
                gt_spec=None, spk_embd=None, infer=True):
        x = self.unit_embed(units)

        if spk_embd is None:
            spk_feat = torch.zeros(x.size(0), x.size(1), self.n_hidden, device=x.device, dtype=x.dtype)
        else:
            if spk_embd.dim() != 2 or spk_embd.size(-1) != self.n_hidden:
                raise ValueError(f"spk_embd shape must be [B, {self.n_hidden}], got {tuple(spk_embd.shape)}")
            spk_vec = spk_embd

            transformed = None
            if hasattr(self, 'spk_transformer') and self.spk_transformer is not None:
                transformed = self.spk_transformer.transformer(spk_vec)
                w0, w1 = 2.0, 1.0
                if hasattr(self, 'spk_transformer_weights'):
                    w = self.spk_transformer_weights
                    if isinstance(w, (list, tuple)) and len(w) == 2:
                        w0, w1 = float(w[0]), float(w[1])
                combined = (w0 * spk_vec + w1 * transformed) / (w0 + w1)
                spk_feat = combined.unsqueeze(1).expand(-1, x.size(1), -1)
            else:
                spk_feat = spk_vec.unsqueeze(1).expand(-1, x.size(1), -1)

        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5) 

        if self.mm_attention is not None:
            f0_log = (1 + f0 / 700).log()
            f0_volume_concat = torch.cat([f0_log, volume], dim=-1)
            f0_volume_embedded = self.f0_volume_embed(f0_volume_concat)
            x, attention_gate = self.mm_attention(x, spk_feat, f0_volume_embedded)
        else:
            f0_log = (1 + f0 / 700).log()
            f0_embedded = self.f0_embed(f0_log)
            volume_embedded = self.volume_embed(volume)
            x = x + spk_feat + f0_embedded + volume_embedded
            attention_gate = torch.tensor(0.0, device=x.device)
        
        if not infer:
            output  = self.decoder(gt_spec,x,infer=False)       
        else:
            output = self.decoder(gt_spec,x,infer=True)

        return output, attention_gate

