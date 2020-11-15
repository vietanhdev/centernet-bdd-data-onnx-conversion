cd src
# train
python main.py ctdet --exp_id coco_resdcn18 --arch resdcn_18  --dataset bdd --batch_size 18 --lr 1e-4 --gpus 0 --num_workers 8
# test
python test.py ctdet --exp_id coco_resdcn18 --arch resdcn_18 --keep_res --resume
# flip test
python test.py ctdet --exp_id coco_resdcn18 --arch resdcn_18 --keep_res --resume --flip_test 
# multi scale test
python test.py ctdet --exp_id coco_resdcn18 --arch resdcn_18 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5
cd ..
