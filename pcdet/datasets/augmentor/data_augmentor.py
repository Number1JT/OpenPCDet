from functools import partial
import numpy as np
from . import augmentor_utils, database_sampler
from ...utils import common_utils


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
   
    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes, points = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE']
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def filter_by_min_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.filter_by_min_points, config=config)
        gt_boxes = data_dict['gt_boxes']
        if gt_boxes.size == 0:  # empty
            return data_dict

        points_num = config['POINTS_NUM']
        assert isinstance(points_num, list)

        points_num_dict = {}
        for name_num in points_num:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            points_num_dict[name] = min_num
        if 'gt_boxes_points_num' in data_dict:
            gt_boxes = data_dict['gt_boxes']
            gt_names = data_dict['gt_names']
            gt_boxes_points_num = data_dict['gt_boxes_points_num']
            keeped = np.zeros_like(gt_boxes_points_num, dtype=np.bool)
            for index, (name, num) in enumerate(zip(gt_names, gt_boxes_points_num)):
                if num > points_num_dict[name]:
                    keeped[index] = True
            data_dict['gt_boxes'] = gt_boxes[keeped]
            data_dict['gt_names'] = gt_names[keeped]
            data_dict['gt_boxes_points_num'] = gt_boxes_points_num[keeped]
            if 'gt_boxes_mask' in data_dict:
                gt_boxes_mask = data_dict['gt_boxes_mask']
                data_dict['gt_boxes_mask'] = gt_boxes_mask[keeped]

        else:
            raise Exception
        return data_dict

    def filter_by_camera_fov(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.filter_by_camera_fov, config=config)

        fov = config['FOV']
        assert isinstance(fov, list)
        fov_neg_angle, fov_pos_angle = fov[0], fov[1]
        fov_neg_cos, fov_pos_cos = np.cos(fov_neg_angle / 180 * np.pi), np.cos(fov_pos_angle / 180 * np.pi)
        # filter points
        points = data_dict['points']
        x, y = points[:, 0], points[:, 1]
        fov_flag = np.logical_or(
            (x / np.linalg.norm((x, y), ord=2, axis=0) > fov_neg_cos) & (y < 0),
            (x / np.linalg.norm((x, y), ord=2, axis=0) > fov_pos_cos) & (y > 0)
        )
        data_dict['points'] = points[fov_flag]
        # filter box
        if 'gt_boxes' in data_dict:
            # filter box which center is not in fov
            gt_boxes = data_dict['gt_boxes']
            if gt_boxes.size == 0:  # empty
                return data_dict
            
            center_x, center_y = gt_boxes[:, 0], gt_boxes[:, 1]
            x, y = center_x, center_y
            fov_flag = np.logical_or(
                (x / np.linalg.norm((x, y), ord=2, axis=0) > fov_neg_cos) & (y < 0),
                (x / np.linalg.norm((x, y), ord=2, axis=0) > fov_pos_cos) & (y > 0)
            )
            data_dict['gt_boxes'] = gt_boxes[fov_flag]
            if 'gt_names' in data_dict:
                gt_names = data_dict['gt_names']
                data_dict['gt_names'] = gt_names[fov_flag]
            if 'gt_boxes_points_num' in data_dict:
                gt_boxes_points_num = data_dict['gt_boxes_points_num']
                data_dict['gt_boxes_points_num'] = gt_boxes_points_num[fov_flag]
            if 'gt_boxes_mask' in data_dict:
                gt_boxes_mask = data_dict['gt_boxes_mask']
                data_dict['gt_boxes_mask'] = gt_boxes_mask[fov_flag]
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)
        if data_dict['gt_boxes'].size > 0:
            data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
                data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
            )
        if 'calib' in data_dict:
            data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            data_dict.pop('gt_boxes_mask')
        return data_dict
