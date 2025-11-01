#!/bin/bash
# ===================================================================
# CoMoSVC 自动训练与关机脚本
# ===================================================================
# 功能：
#   1. 启动训练任务
#   2. 训练完成后自动关机
#   3. 支持异常处理
# ===================================================================

# === 环境设置 ===
CONFIG_PATH="configs/diffusion.yaml"   # 配置文件路径
LOG_PATH="/root/autodl-tmp/CoMoSVC/train.log"  # 日志输出路径
DATE_STR=$(date +"%Y-%m-%d_%H-%M-%S")
COMPLETION_FLAG="/root/autodl-tmp/train_completed_$DATE_STR.flag"

# === 激活环境（可选）===
# 如果你有conda环境，请取消下面注释
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your_env_name

# === 进入项目目录 ===
cd /root/autodl-tmp/CoMoSVC || {
    echo "❌ 无法进入项目目录" | tee -a $LOG_PATH
    exit 1
}

echo "========================================" | tee -a $LOG_PATH
echo "🚀 [$(date)] 开始训练任务..." | tee -a $LOG_PATH
echo "📝 配置文件: $CONFIG_PATH" | tee -a $LOG_PATH
echo "📄 日志文件: $LOG_PATH" | tee -a $LOG_PATH
echo "========================================" | tee -a $LOG_PATH
echo "" | tee -a $LOG_PATH

# === 启动训练（前台运行，等待完成）===
torchrun --nproc_per_node=2 train.py -c $CONFIG_PATH 2>&1 | tee -a $LOG_PATH

# === 获取训练退出码 ===
TRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a $LOG_PATH
echo "========================================" | tee -a $LOG_PATH
echo "📊 训练进程退出码: $TRAIN_EXIT_CODE" | tee -a $LOG_PATH
echo "========================================" | tee -a $LOG_PATH

# === 判断训练结果 ===
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    # ✅ 训练正常完成 - 自动关机
    echo "" | tee -a $LOG_PATH
    echo "========================================" | tee -a $LOG_PATH
    echo "✅ [$(date)] 训练成功完成！" | tee -a $LOG_PATH
    echo "========================================" | tee -a $LOG_PATH
    echo "$DATE_STR: Training completed successfully" > $COMPLETION_FLAG
    
    # 显示最终模型信息
    echo "" | tee -a $LOG_PATH
    echo "📦 已保存的模型:" | tee -a $LOG_PATH
    ls -lh logs/model_*.pt 2>/dev/null | tail -5 | tee -a $LOG_PATH
    
    echo "" | tee -a $LOG_PATH
    echo "⏰ 训练正常完成，系统将在 60 秒后自动关机..." | tee -a $LOG_PATH
    echo "💡 如需取消关机，请执行: shutdown -c" | tee -a $LOG_PATH
    echo "" | tee -a $LOG_PATH
    
    # 倒计时
    for i in {60..1}; do
        echo -ne "\r⏳ 关机倒计时: $i 秒   "
        sleep 1
    done
    echo ""
    
    echo "🔌 [$(date)] 执行关机命令..." | tee -a $LOG_PATH
    shutdown -h now
    
else
    # ❌ 训练异常退出 - 不关机，保留现场
    echo "" | tee -a $LOG_PATH
    echo "========================================" | tee -a $LOG_PATH
    echo "❌ [$(date)] 训练异常退出！" | tee -a $LOG_PATH
    echo "📊 退出码: $TRAIN_EXIT_CODE" | tee -a $LOG_PATH
    echo "========================================" | tee -a $LOG_PATH
    echo "$DATE_STR: Training failed with exit code $TRAIN_EXIT_CODE" > $COMPLETION_FLAG
    
    # 显示最后的日志
    echo "" | tee -a $LOG_PATH
    echo "📋 最后20行训练日志:" | tee -a $LOG_PATH
    echo "----------------------------------------" | tee -a $LOG_PATH
    tail -20 $LOG_PATH
    echo "----------------------------------------" | tee -a $LOG_PATH
    
    echo "" | tee -a $LOG_PATH
    echo "⚠️  训练异常退出，系统不会自动关机" | tee -a $LOG_PATH
    echo "💡 请检查日志文件排查问题: $LOG_PATH" | tee -a $LOG_PATH
    echo "📂 检查点文件位置: logs/" | tee -a $LOG_PATH
    echo "🔍 可能的错误原因:" | tee -a $LOG_PATH
    echo "   - NaN loss (数值不稳定)" | tee -a $LOG_PATH
    echo "   - OOM (显存不足)" | tee -a $LOG_PATH
    echo "   - 数据加载错误" | tee -a $LOG_PATH
    echo "   - 配置错误" | tee -a $LOG_PATH
    echo "" | tee -a $LOG_PATH
    echo "🛠️  修复后可重新运行: ./train_with_auto_shutdown.sh" | tee -a $LOG_PATH
    echo "" | tee -a $LOG_PATH
    
    # 不关机，保持运行状态供调试
    echo "✋ 系统保持运行状态，等待人工处理..." | tee -a $LOG_PATH
    exit 1
fi

