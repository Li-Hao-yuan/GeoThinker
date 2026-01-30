set -e
export LMMS_EVAL_LAUNCHER="accelerate"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NVLS_ENABLE=0

output_path=logs/$(TZ="Asia/Shanghai" date "+%Y%m%d")
model_path=GeoThinker-8B

benchmark=vsibench

accelerate launch --num_processes=8 --main_process_port 29505 -m lmms_eval \
    --model geothinker \
    --model_args pretrained=$model_path,use_flash_attention_2=true,max_num_frames=32,max_length=12800,max_pixels=451584,min_pixels=12544 \
    --tasks ${benchmark} \
    --batch_size 1 \
    --output_path $output_path