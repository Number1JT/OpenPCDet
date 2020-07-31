from pcdet.datasets.waymo.waymo_dataset import create_waymo_infos,check_dataset,debug_data

if __name__ == '__main__':
    import sys
    argv = [1, 'create_waymo_infos', 'tools/cfgs/dataset_configs/waymo_dataset.yaml']
    if True:
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()

        if False:
            create_waymo_infos(
                dataset_cfg=dataset_cfg,
                class_names= ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST'],
                data_path=ROOT_DIR / 'data' / 'waymo',
                save_path=ROOT_DIR / 'data' / 'waymo',
                workers=8
            )

        if True:
            check_dataset(
                dataset_cfg=dataset_cfg,
                class_names=['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST'],
                skip_index=0,
                training=True,
                split='train',
                root_path=ROOT_DIR / 'data' / 'waymo',
                logger=None
            )

        if False:
            debug_data(
                dataset_cfg=dataset_cfg,
                class_names=['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST'],
                data_index=0,
                training=False,
                split='valid',
                root_path=ROOT_DIR / 'data' / 'waymo',
                logger=None
            )
