import os
import torch
import torch.nn as nn
import yaml
from Vocoder import Vocoder
from como import Como
# from speaker_encoder import SpeakerEncoder  # DEPRECATED: Now using spk_embd_transformer
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
    
    # load model - 确保与训练时完全一致的架构
    use_attention = getattr(args.model, 'use_attention', True)
    print(f"🔧 Model Architecture Settings:")
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
                attention=use_attention,  # 使用配置文件中的attention设置
                config=args  # 传递配置参数
                )
    
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    
    # 检查checkpoint中的参数
    print("🔍 Checking checkpoint parameters...")
    ckpt_keys = set(ckpt['model'].keys())
    model_keys = set(model.state_dict().keys())
    
    print(f"📋 Checkpoint has {len(ckpt_keys)} parameters")
    print(f"📋 Model expects {len(model_keys)} parameters")
    
    # 检查注意力机制相关参数
    attention_keys = [k for k in ckpt_keys if 'mm_attention' in k or 'gate_alpha_raw' in k]
    if attention_keys:
        print(f"🎯 Found attention parameters in checkpoint: {attention_keys}")
        for key in attention_keys:
            if 'gate_alpha_raw' in key:
                print(f"   {key}: {ckpt['model'][key].item():.6f}")
    else:
        print("❌ No attention parameters found in checkpoint!")
    
    # 加载模型参数
    missing_keys, unexpected_keys = model.load_state_dict(ckpt['model'], strict=False)
    
    if missing_keys:
        print(f"⚠️  Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"⚠️  Unexpected keys: {unexpected_keys}")
    
    # 验证gate_alpha_raw是否正确加载
    if hasattr(model, 'mm_attention') and model.mm_attention is not None:
        gate_value = torch.sigmoid(model.mm_attention.gate_alpha_raw).item()
        print(f"🎛️  Loaded gate_alpha_raw value: {gate_value:.6f}")
    
    model.eval()
    return model, vocoder, args


class ComoSVC(nn.Module):
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
            config=None  # 新增配置参数
            ):
        super().__init__()

        self.unit_embed = nn.Linear(input_channel, n_hidden)
        
        # 非注意力机制：分别处理f0和volume
        self.f0_embed = nn.Linear(1, n_hidden)
        self.volume_embed = nn.Linear(1, n_hidden)
        
        # 注意力机制：f0和volume拼接后线性变换
        self.f0_volume_embed = nn.Linear(2, n_hidden)  # 2维输入：f0 + volume

        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        else:
            self.aug_shift_embed = None
        
        # Speaker Encoder (DEPRECATED: 仅用于辅助任务，现在使用spk_embd_transformer)
        # 从配置文件读取参数，如果没有则使用默认值
        # if config and hasattr(config, 'model') and hasattr(config.model, 'speaker_encoder'):
        #     spk_config = config.model.speaker_encoder
        #     self.spk_encoder = SpeakerEncoder(
        #         c_in=spk_config.get('c_in', 80),
        #         c_h=spk_config.get('c_h', n_hidden),
        #         c_out=spk_config.get('c_out', n_hidden),
        #         kernel_size=spk_config.get('kernel_size', 5),
        #         bank_size=spk_config.get('bank_size', 8),
        #         bank_scale=spk_config.get('bank_scale', 1),
        #         c_bank=spk_config.get('c_bank', n_hidden),
        #         n_conv_blocks=spk_config.get('n_conv_blocks', 6),
        #         n_dense_blocks=spk_config.get('n_dense_blocks', 2),
        #         subsample=spk_config.get('subsample', [1,2,1,2,1,2]),
        #         act=spk_config.get('act', "lrelu"),
        #         dropout_rate=spk_config.get('dropout_rate', 0.1),
        #     )
        # else:
        #     # 使用默认参数
        #     self.spk_encoder = SpeakerEncoder(
        #         c_in=80,         # mel 维度
        #         c_h=n_hidden,    # 与模型 hidden 对齐
        #         c_out=n_hidden,  # 输出维度
        #         kernel_size=5,
        #         bank_size=8,
        #         bank_scale=1,
        #         c_bank=n_hidden,
        #         n_conv_blocks=6,
        #         n_dense_blocks=2,
        #         subsample=[1,2,1,2,1,2],
        #         act="lrelu",
        #         dropout_rate=0.1,
        #     )
        
        # 保留spk_encoder属性为None，用于兼容性
        self.spk_encoder = None
        
        self.n_hidden = n_hidden
        self.decoder = Como(out_dims, n_layers, n_chans, n_hidden, total_steps) 
        self.input_channel = input_channel

        if attention:
            # 从配置文件读取注意力机制参数
            if config and hasattr(config, 'model') and hasattr(config.model, 'attention'):
                attn_config = config.model.attention
                self.mm_attention = MultiModalCrossAttention(
                    d_model=n_hidden,
                    num_heads=attn_config.get('num_heads', 8),
                    dropout=attn_config.get('dropout', 0.1),
                    init_alpha=attn_config.get('init_alpha', 1.0)  # 门控固定为1.0
                )
            else:
                # 使用默认参数
                self.mm_attention = MultiModalCrossAttention(
                    d_model=n_hidden,
                    num_heads=8,
                    dropout=0.1,
                    init_alpha=1.0  # 门控固定为1.0
                )
        else:
            self.mm_attention = None

        # 可选的说话人嵌入变换组件（通过yaml控制启用）
        self.spk_transformer = None
        self.spk_transformer_weights = None
        if config and hasattr(config, 'model') and hasattr(config.model, 'spk_embd_transformer'):
            tcfg = config.model.spk_embd_transformer
            if tcfg.get('enabled', False):
                self.spk_transformer = SpeakerEmbeddingTransformerWithGRL(
                    spk_embd_dim=n_hidden,
                    output_dim=n_hidden,
                    f0_pred_dim=1,  # 固定为1，实际使用F0分布分类
                    transform_type=tcfg.get('transform_type', 'linear'),
                    transform_config=tcfg.get('transform_config', {}),
                    dropout_rate=0.1,
                )
                self.spk_transformer_weights = tcfg.get('combine_weights', [1, 2])
        

    def forward(self, units, f0, volume, ref_mel=None, speaker_id=None, aug_shift=None,
                gt_spec=None, spk_embd=None, infer=True):
          
        '''
        input: 
            units: B x n_frames x n_unit
            ref_mel: B x 80 x T_ref (参考音频的mel谱，用于speaker encoder，现在可选)
            speaker_id: B (speaker ID，仅在few-shot模式下需要)
            gt_spec: B x 80 x T (目标mel谱，用于重建损失)
            spk_embd: B x n_hidden (预生成的说话人嵌入，优先使用)
        return: 
            dict of B x n_frames x feat
        '''

        # 内容特征嵌入
        x = self.unit_embed(units)  # [B, T, n_hidden]

        # 说话人特征处理（完全由预生成的spk_embd提供，不再调用spk_encoder提取）
        if spk_embd is None:
            # 若未提供，则退化为零向量（不再回退到spk_encoder计算）
            spk_feat = torch.zeros(x.size(0), x.size(1), self.n_hidden, device=x.device, dtype=x.dtype)
        else:
            # 校验维度
            if spk_embd.dim() != 2 or spk_embd.size(-1) != self.n_hidden:
                raise ValueError(f"spk_embd shape must be [B, {self.n_hidden}], got {tuple(spk_embd.shape)}")
            spk_vec = spk_embd  # [B, n_hidden]

            # 可选：通过spk_embd_transformer进行变换，并与原始嵌入按(1:2)加权
            transformed = None
            if hasattr(self, 'spk_transformer') and self.spk_transformer is not None:
                # 仅返回变换后的嵌入供下游使用；F0对抗在solver内计算
                transformed = self.spk_transformer.transformer(spk_vec)  # [B, n_hidden]
                w0, w1 = 2.0, 1.0
                if hasattr(self, 'spk_transformer_weights'):
                    w = self.spk_transformer_weights
                    if isinstance(w, (list, tuple)) and len(w) == 2:
                        w0, w1 = float(w[0]), float(w[1])
                combined = (w0 * spk_vec + w1 * transformed) / (w0 + w1)
                spk_feat = combined.unsqueeze(1).expand(-1, x.size(1), -1)
            else:
                spk_feat = spk_vec.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, T, n_hidden]

        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5) 

        if self.mm_attention is not None:
            # 注意力机制：f0 log变换后与volume拼接
            f0_log = (1 + f0 / 700).log()  # f0 log变换
            f0_volume_concat = torch.cat([f0_log, volume], dim=-1)  # [B, T, 2]
            f0_volume_embedded = self.f0_volume_embed(f0_volume_concat)  # [B, T, n_hidden]
            x, attention_gate = self.mm_attention(x, spk_feat, f0_volume_embedded)
        else:
            # 非注意力机制：f0 log变换后分别处理
            f0_log = (1 + f0 / 700).log()  # f0 log变换
            f0_embedded = self.f0_embed(f0_log)  # [B, T, n_hidden]
            volume_embedded = self.volume_embed(volume)  # [B, T, n_hidden]
            x = x + spk_feat + f0_embedded + volume_embedded
            attention_gate = torch.tensor(0.0, device=x.device)  # 非注意力模式下门控因子为0
        
        if not infer:
            output  = self.decoder(gt_spec,x,infer=False)       
        else:
            output = self.decoder(gt_spec,x,infer=True)

        return output, attention_gate

