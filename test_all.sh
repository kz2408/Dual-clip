# # 零样本 CLIP（不加载 ckpt）
python coco_test.py --model-type clip --dataset voc-lt --data ./data/voc --lr 1e-5 --loss_function dbl --topk 16 --alpha 0.4 --resume checks/clip_80.346_e80.ckpt

# # CoOp
python coco_test.py --model-type coop --dataset voc-lt --data ./data/voc --lr 1e-5 --loss_function dbl --topk 16 --alpha 0.4 --resume coop_voc-lt_coop_best_79.602_e80.ckpt

# # ViT + 模板
python coco_test.py --model-type clip_vit --dataset voc-lt --data ./data/voc --lr 1e-5 --loss_function dbl --topk 16 --alpha 0.4  --resume clipvit_base_vit_voc-lt__best_92.775_e21.ckpt

# # 双视图 + 可学习 prompt
python coco_test.py --model-type clip_vit_dual --dataset voc-lt --data ./data/voc --lr 1e-5 --loss_function dbl --topk 16 --alpha 0.4  --resume clipdual_voc-lt__best_85.222_e6.ckpt


