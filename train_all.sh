# # 零样本 / CoOp（coco_runner）
# python coco_runner.py --model-type clip --dataset voc-lt --data ./data/voc --lr 1e-5 --loss_function dbl --topk 16 --alpha 0.4 --resume ""
# python coco_runner.py --model-type coop --dataset voc-lt --data ./data/voc --lr 1e-5 --loss_function dbl --topk 16 --alpha 0.4 --resume ""
# python coco_runner_dual.py --model-type clip_vit_dual --dataset voc-lt --data ./data/voc --lr 1e-5 --loss_function dbl --topk 16 --alpha 0.4 --resume ""

# # ViT+模板 / 双视图（coco_runner_dual）
# python coco_runner_dual.py --model-type clip_vit --dataset voc-lt --data ./data/voc --lr 1e-5 --loss_function dbl --topk 16 --alpha 0.4 --Dy_en 0 --Cross_en 0 --use_dynamic_prompt 1 --resume "" 
python coco_runner_dual.py --model-type clip_vit --dataset voc-lt --data ./data/voc --lr 1e-5 --loss_function dbl --topk 16 --alpha 0.4 --use_dynamic_prompt 1 --use_text_enhance 1 --use_dual_consistency 1 --resume "" 
