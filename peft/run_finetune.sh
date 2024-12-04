export TF_CPP_MIN_LOG_LEVEL=2 

timestamp=$(date +"%m%d%H%M%S")
log_dir='../log'
if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi
export log_file="${log_dir}/logfile_${timestamp}.log"
if [ -f "$log_file" ]; then
    rm -f "$log_file"
fi

nohup python3 finetune.py \
    --checkpoint-path='/root/autodl-tmp/timesfm-1.0-200m/checkpoints' \
    --checkpoint-dir="/root/autodl-tmp/ckpt/16" \
    --backend="gpu" \
    --horizon-len=96 \
    --context-len=96 \
    --freq="15min" \
    --data-path="/root/autodl-tmp/timesfm/datasets/gnrx/weather_long_power_dropnull.csv" \
    --batch-size=16 \
    --num-epochs=500 \
    --learning-rate=1e-3 \
    --adam-epsilon=1e-7 \
    --adam-clip-threshold=1e2 \
    --early-stop-patience=10 \
    --datetime-col="data_time" \
    --cos-initial-decay-value=1e-4 \
    --cos-decay-steps=40000 \
    --cos-final-decay-value=1e-5 \
    --ema-decay=0.9999 \
    > $log_file 2>&1 &


# sleep 2
# tail -f $log_file | grep -E "Epoch|Loss|patience"
# tail -f logfile_11251756.log |  grep --color -E "Epoch|Loss|patience"
    # --use-lora \
    # --lora-rank=1 \
    # --lora-target-modules="all" \
    # --use-dora \
    # /root/autodl-tmp/timesfm-1.0-200m/checkpoints/