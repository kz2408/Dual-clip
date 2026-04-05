
echo "------------------------------------------------------"
echo "    Start detecting available GPUs    "
echo "------------------------------------------------------"
gpu_num=0
for i in $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); do
  gpu_num=$((gpu_num + 1))
done
num=0
for i in $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); do
  if [ $i -lt 14000 ]; then
    echo '    GPU ' $num ' is available'
    echo "------------------------------------------------------"
    break
  fi
  num=$((num + 1))
done
if [ $num -ge $gpu_num ]; then
  echo '    No GPU is available, what to do next :'
  echo "------------------------------------------------------"
  echo "    1. Waiting until GPU is available."
  echo "    2. Choosing a GPU manually."
  echo "    3. exit."
  option=-1
  read option
  if [ $option -eq 1 ]; then
    echo "------------------------------------------------------"
    echo "    Start waiting............................."
    echo "------------------------------------------------------"
    flag=0
    while [ $flag -ne 1 ]
    do
      num=0
      for i in $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); do
        if [ $i -lt 500 ]; then
          echo '    GPU ' $num ' is available'
          flag=1
          echo '    Start running.............................'
          break
        else
          currTime='----Waiting: '$(date +"%Y-%m-%d %T")
          echo $currTime
          sleep 1
        fi
        num=$((num + 1))
      done
    done
  elif [ $option -eq 2 ]; then
      echo "------------------------------------------------------"
      echo "    Please choose a number in [0~"$((gpu_num-1))"]"
      read num
      echo '    GPU ' $num ' is chosen'
  else
    exit
  fi
fi
echo "------------------------------------------------------"

# CUDA_VISIBLE_DEVICES=$num python coco_runner.py --dataset coco-lt --lr 1e-5 --loss_function bce --topk 32 --alpha 0.4
# CUDA_VISIBLE_DEVICES=$num python fsl_runner.py --dataset voc --lr 1e-5 --topk 32 --alpha 0.4
# CUDA_VISIBLE_DEVICES=$num python fsl_eval.py --dataset voc --resume /data2/yanjiexuan/checkpoints/RC-Tran/MKT_fs_checkpoint/ema_fsl_First_Stage_al0.4_lr1e-05_clip_vit_topk32_voc_best_95.796_e15.ckpt --lr 1e-5 --shot 5
CUDA_VISIBLE_DEVICES=$num python coco_test.py --resume /data2/yanjiexuan/checkpoints/RC-Tran/MKT_LT_checkpoint/ema_First_Stage_al0.4_lr1e-05_dbl_clip_vit_topk32_coco-lt_best_73.347_e38.ckpt --dataset coco-lt