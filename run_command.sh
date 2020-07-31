generate dataset:
    cd tools/
    CUDA_VISIBLE_DEVICES=1 python -m pcdet.datasets.waymo.waymo_dataset create_waymo_dataset cfgs/dataset_configs/waymo_dataset.yaml
    # optional
    # check dataset
    CUDA_VISIBLE_DEVICES=1 python -m pcdet.datasets.waymo.waymo_dataset check_dataset cfgs/dataset_configs/waymo_dataset.yaml
    # generate/check/debug also in generate_dataset_debug.py
    CUDA_VISIBLE_DEVICES=1 python -m generate_dataset_debug.py

train:
    cd tools/
    # single gpu
    CUDA_VISIBLE_DEVICES=7 python -m ipdb train.py --cfg_file  cfgs/waymo_models/pv_rcnn.yaml --batch_size 2 --epochs 100 --workers 1
    # multi gpu
    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=6 train.py --launcher pytorch --cfg_file  cfgs/waymo_models/pv_rcnn.yaml --batch_size 18 --epochs 50 --workers 18 --sync_bn


test:
    cd tools/
    # single gpu
    CUDA_VISIBLE_DEVICES=7 python test.py --cfg_file cfgs/waymo_models/pv_rcnn.yaml --batch_size 1 --ckpt ../output/waymo_models/pv_rcnn/default/ckpt/checkpoint_epoch_6.pth --split valid --save_to_file --extra_tag 2020-07-20-14-38
    # multi gpu
    CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 test.py --launcher pytorch --cfg_file cfgs/waymo_models/pv_rcnn.yaml --batch_size 32 --workers 4 --ckpt ../output/waymo_models/pv_rcnn/default/ckpt/checkpoint_epoch_11.pth --split valid --save_to_file --extra_tag 2020-07-20-14-38
