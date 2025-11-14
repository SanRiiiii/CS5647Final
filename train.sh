#!/bin/bash

CONFIG_PATH="configs/diffusion.yaml"
LOG_PATH="/root/autodl-tmp/Project/train.log"
DATE_STR=$(date +"%Y-%m-%d_%H-%M-%S")
COMPLETION_FLAG="/root/autodl-tmp/train_completed_$DATE_STR.flag"

cd /root/autodl-tmp/Project || {
    echo "Unable to enter project directory" | tee -a $LOG_PATH
    exit 1
}

echo "========================================" | tee -a $LOG_PATH
echo "[$(date)] Starting training..." | tee -a $LOG_PATH
echo "Config: $CONFIG_PATH" | tee -a $LOG_PATH
echo "Log: $LOG_PATH" | tee -a $LOG_PATH
echo "========================================" | tee -a $LOG_PATH
echo "" | tee -a $LOG_PATH

torchrun --nproc_per_node=2 train.py -c $CONFIG_PATH 2>&1 | tee -a $LOG_PATH

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a $LOG_PATH
echo "========================================" | tee -a $LOG_PATH
echo "Training exit code: $TRAIN_EXIT_CODE" | tee -a $LOG_PATH
echo "========================================" | tee -a $LOG_PATH

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "" | tee -a $LOG_PATH
    echo "========================================" | tee -a $LOG_PATH
    echo "[$(date)] Training completed successfully" | tee -a $LOG_PATH
    echo "========================================" | tee -a $LOG_PATH
    echo "$DATE_STR: Training completed successfully" > $COMPLETION_FLAG
    
    echo "" | tee -a $LOG_PATH
    echo "Saved models:" | tee -a $LOG_PATH
    ls -lh logs/model_*.pt 2>/dev/null | tail -5 | tee -a $LOG_PATH
    
    echo "" | tee -a $LOG_PATH
    echo "System will shutdown in 60 seconds..." | tee -a $LOG_PATH
    echo "To cancel shutdown, execute: shutdown -c" | tee -a $LOG_PATH
    echo "" | tee -a $LOG_PATH
    
    for i in {60..1}; do
        echo -ne "\rShutdown countdown: $i seconds   "
        sleep 1
    done
    echo ""
    
    echo "[$(date)] Executing shutdown..." | tee -a $LOG_PATH
    shutdown -h now
    
else
    echo "" | tee -a $LOG_PATH
    echo "========================================" | tee -a $LOG_PATH
    echo "[$(date)] Training failed" | tee -a $LOG_PATH
    echo "Exit code: $TRAIN_EXIT_CODE" | tee -a $LOG_PATH
    echo "========================================" | tee -a $LOG_PATH
    echo "$DATE_STR: Training failed with exit code $TRAIN_EXIT_CODE" > $COMPLETION_FLAG
    
    echo "" | tee -a $LOG_PATH
    echo "Last 20 lines of training log:" | tee -a $LOG_PATH
    echo "----------------------------------------" | tee -a $LOG_PATH
    tail -20 $LOG_PATH
    echo "----------------------------------------" | tee -a $LOG_PATH
    
    echo "" | tee -a $LOG_PATH
    echo "Training failed, system will not shutdown" | tee -a $LOG_PATH
    echo "Check log file: $LOG_PATH" | tee -a $LOG_PATH
    echo "Checkpoint location: logs/" | tee -a $LOG_PATH
    echo "Possible error causes:" | tee -a $LOG_PATH
    echo "   - NaN loss" | tee -a $LOG_PATH
    echo "   - OOM" | tee -a $LOG_PATH
    echo "   - Data loading error" | tee -a $LOG_PATH
    echo "   - Configuration error" | tee -a $LOG_PATH
    echo "" | tee -a $LOG_PATH
    echo "Re-run after fixing: ./train_with_auto_shutdown.sh" | tee -a $LOG_PATH
    echo "" | tee -a $LOG_PATH
    
    echo "System remains running for debugging..." | tee -a $LOG_PATH
    exit 1
fi

