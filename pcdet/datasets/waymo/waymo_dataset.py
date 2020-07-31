import os
import pickle
import copy
import numpy as np
from skimage import io
from pathlib import Path
import tqdm
import yaml
from easydict import EasyDict
from collections import ChainMap
import random
import tensorflow as tf
from waymo_open_dataset import label_pb2
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import metrics_pb2

from ...utils import box_utils, common_utils, calibration_kitti, object3d_kitti
from ..dataset import DatasetTemplate
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from .frame_parser import FrameParser


class WaymoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, split='train', root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[split]
        self.root_split_path = self.root_path / self.split /'sample'

        split_file= self.root_path / self.split / 'sample' / 'dataset.bin'
        if split_file.exists():
            with open(split_file, 'rb') as f:
                datasets = pickle.load(f)
            self.sample_id_list = [x.strip('.bin') for x in datasets]
        else:
            self.sample_id_list = None
        self.waymo_infos = []
        self.include_waymo_data(split)

        # used for generate commit file for waymo website
        self.context_infos_map = {}  # key: frame_id value: context_name and frame_timestamp_micros
        context_infos_file = self.root_path / self.split / 'sample' / 'context_infos.bin'
        if context_infos_file.exists():
            with open(context_infos_file, 'rb') as f:
                self.context_infos_map = pickle.load(f)
        else:
            self.context_infos_map = None


    def include_waymo_data(self, split):
        if self.logger is not None:
            self.logger.info('Loading WAYMO dataset')
        waymo_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[split]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                waymo_infos.extend(infos)

        self.waymo_infos.extend(waymo_infos)

        if self.logger is not None:
            self.logger.info('Total samples for WAYMO dataset: %d' % (len(waymo_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[split]
        self.root_split_path = self.root_path / self.split /'sample'

        split_file= self.root_path / self.split / 'sample' / 'dataset.bin'
        with open(split_file, 'rb') as f:
            datasets = pickle.load(f)
        self.sample_id_list = [x.strip('.bin') for x in datasets] if split_file.exists() else None

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'laser' / ('%s.bin' % idx)
        assert lidar_file.exists()
        with open(lidar_file, 'rb') as f:
            lidar = pickle.load(f)
        return lidar

    def get_label(self, idx):
        label_file = self.root_split_path / 'label' / ('%s.bin' % idx)
        assert label_file.exists()
        with open(label_file, 'rb') as f:
            label = pickle.load(f)
        return label

    def get_road_plane(self, idx):
        return None

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s frame_name: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': len(self.dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list), 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            if has_label:
                info['annos'] = self.get_label(sample_idx)
            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        # process_single_scene(sample_id_list[0]) # DEBUG 
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        _split = self.dataset_cfg.DATA_SPLIT[split]
        database_save_path = Path(self.root_path) / ('gt_database' if _split == 'train' else ('gt_database_%s' % _split))
        db_info_save_path = Path(self.root_path) / ('waymo_dbinfos_%s.pkl' % _split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']

            laser_box3d, laser_box3d_categorys, detection_difficulty_level, num_lidar_points_in_box = FrameParser.parse_laser_labels(annos)
            names = laser_box3d_categorys
            difficulty = detection_difficulty_level
            gt_boxes = laser_box3d

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]
                gt_points[:, :3] -= gt_boxes[i, :3]

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {
                        'name': names[i],'difficulty': difficulty[i],
                        'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],'gt_points':gt_points,
                    }
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None, save_laser=False):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        assert self.context_infos_map is not None

        # https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/protos/metrics.proto
        objects = metrics_pb2.Objects()
        for index, box_dict in enumerate(pred_dicts):
            # https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto
            annos_in_Label_format = []

            frame_id = batch_dict['frame_id'][index]
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()

            for pred_score, pred_box, pred_label in zip(pred_scores, pred_boxes, pred_labels):
                _box = label_pb2.Label.Box()
                _box.center_x = pred_box[0]
                _box.center_y = pred_box[1]
                _box.center_z = pred_box[2]
                _box.length = pred_box[3]
                _box.width = pred_box[4]
                _box.height = pred_box[5]
                _box.heading = pred_box[6]
                _type = label_pb2.Label.Type.Name(pred_label)
                _score = pred_score

                """Creates a prediction objects file in label_pb2.Label format."""
                # format
                anno_in_Label_format = label_pb2.Label(box=_box, type=_type)
                annos_in_Label_format.append(anno_in_Label_format)

                """Creates a prediction objects file in metrics_pb2.Objects."""
                anno_in_Object_format = metrics_pb2.Object()
                # The following 3 fields are used to uniquely identify a frame a prediction
                # is predicted at. Make sure you set them to values exactly the same as what
                # we provided in the raw data. Otherwise your prediction is considered as a
                # false negative.
                context_name = self.context_infos_map[frame_id]['context_name']
                anno_in_Object_format.context_name = context_name
                # The frame timestamp for the prediction. See Frame::timestamp_micros in
                # dataset.proto.
                invalid_ts = self.context_infos_map[frame_id]['frame_timestamp_micros']
                anno_in_Object_format.frame_timestamp_micros = invalid_ts
                # This is only needed for 2D detection or tracking tasks.
                # Set it to the camera name the prediction is for.
                # metrics_pb2Object_format.camera_name = dataset_pb2.CameraName.FRONT
                # Populating box and score.
                anno_in_Object_format.object.CopyFrom(anno_in_Label_format)
                # This must be within [0.0, 1.0]. It is better to filter those boxes with
                # small scores to speed up metrics computation.
                anno_in_Object_format.score = _score
                # For tracking, this must be set and it must be unique for each tracked
                # sequence.
                # anno_in_Object_format.object.id = 'unique object tracking ID'
                objects.objects.append(anno_in_Object_format)

            if output_path is not None:
                output_path.mkdir(parents=True, exist_ok=True)
                cur_det_file = output_path / ('%s.bin' % frame_id)
                with open(cur_det_file, 'wb') as f:
                    pickle.dump(annos_in_Label_format, f)
                # save laser points
                if save_laser:
                    points = batch_dict['points'].cpu().numpy()
                    points = points[points[:, 0] == index]
                    points = points[:, 1:]
                    _output_path = output_path / '../laser/'
                    _output_path.mkdir(parents=True, exist_ok=True)
                    cur_laser_file = _output_path / ('%s.bin' % frame_id)
                    with open(cur_laser_file, 'wb') as f:
                        pickle.dump(points, f)
        return objects

    # def evaluation(self, det_annos, class_names, **kwargs):
    #     if 'annos' not in self.kitti_infos[0].keys():
    #         return None, {}
    #
    #     from .kitti_object_eval_python import eval as kitti_eval
    #
    #     eval_det_annos = copy.deepcopy(det_annos)
    #     eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
    #     ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)
    #
    #     return ap_result_str, ap_dict

    @staticmethod
    def cal_mAP(prediction_file, gt_file, ignore_box_by_point_num=-1, fov_only=False, fov=[-115.2, 115.2]):
        import tensorflow as tf
        from google.protobuf import text_format
        from waymo_open_dataset import label_pb2
        from waymo_open_dataset import dataset_pb2
        from waymo_open_dataset.protos import metrics_pb2
        from waymo_open_dataset.metrics.python import detection_metrics

        streams = open(prediction_file, 'rb').read()
        objects = metrics_pb2.Objects()
        objects.ParseFromString(bytearray(streams))
        pred_objects = objects

        streams = open(gt_file, 'rb').read()
        objects = metrics_pb2.Objects()
        objects.ParseFromString(bytearray(streams))
        gt_objects = objects
        no_label_zone_objects = gt_objects.no_label_zone_objects

        def set_config():
            config = metrics_pb2.Config()
            config_text = """
            num_desired_score_cutoffs: 11
            breakdown_generator_ids: OBJECT_TYPE
            difficulties {
            }
            matcher_type: TYPE_HUNGARIAN
            iou_thresholds: 0.5
            iou_thresholds: 0.5
            iou_thresholds: 0.5
            iou_thresholds: 0.5
            iou_thresholds: 0.5
            box_type: TYPE_3D
            """
            text_format.Merge(config_text, config)
            return config

        def check_in_no_label_zone():
            return False

        def filter_by_camera_fov(center_x, center_y):
            assert isinstance(fov, list)
            fov_neg_angle, fov_pos_angle = fov[0], fov[1]
            fov_neg_cos, fov_pos_cos = np.cos(fov_neg_angle / 180 * np.pi), np.cos(fov_pos_angle / 180 * np.pi)
            x, y = center_x, center_y
            fov_flag = np.logical_or(
                (x / np.linalg.norm((x, y), ord=2, axis=0) > fov_neg_cos) & (y < 0),
                (x / np.linalg.norm((x, y), ord=2, axis=0) > fov_pos_cos) & (y > 0)
            )
            return fov_flag

        config = set_config()

        frame_ids_dict = {} # key:frame_name value:frame_id
        prediction_frame_id = []
        prediction_bbox = []
        prediction_type = []
        prediction_score = []
        prediction_overlap_nlz = []

        for _object in pred_objects.objects:
            context_name = _object.context_name
            frame_timestamp_micros = _object.frame_timestamp_micros
            label = _object.object
            score = _object.score
            frame_id = '%s_%s' % (context_name, frame_timestamp_micros)
            if frame_id not in frame_ids_dict:
                frame_ids_dict[frame_id] = len(frame_ids_dict) + 1
            prediction_frame_id.append(frame_ids_dict[frame_id])

            prediction_bbox.append(
                [
                    label.box.center_x,label.box.center_y,label.box.center_z,
                    label.box.length,label.box.width,label.box.height,
                    label.box.heading]
            )
            prediction_type.append(label.type)
            prediction_score.append(score)
            prediction_overlap_nlz.append(
                check_in_no_label_zone()
            )

        ground_truth_frame_id = []
        ground_truth_bbox = []
        ground_truth_type = []
        ground_truth_difficulty = []
        for _object in gt_objects.objects:
            context_name = _object.context_name
            frame_timestamp_micros = _object.frame_timestamp_micros
            label = _object.object
            frame_id = '%s_%s' % (context_name, frame_timestamp_micros)
            num_lidar_points_in_box = label.num_lidar_points_in_box
            # detection_difficulty_level = _object.detection_difficulty_level
            detection_difficulty_level = 0
            if frame_id not in frame_ids_dict:
                continue
            if num_lidar_points_in_box < ignore_box_by_point_num:
                continue
            in_fov = True
            if fov_only:
                in_fov = filter_by_camera_fov(center_x=label.box.center_x, center_y=label.box.center_y)
            if not in_fov:
                continue

            ground_truth_frame_id.append(frame_ids_dict[frame_id])
            ground_truth_bbox.append(
                [
                    label.box.center_x, label.box.center_y, label.box.center_z,
                    label.box.length,label.box.width,label.box.height,
                    label.box.heading]
            )
            ground_truth_type.append(label.type)
            ground_truth_difficulty.append(detection_difficulty_level)

        prediction_frame_id = tf.convert_to_tensor(prediction_frame_id, dtype=tf.int64)
        prediction_bbox = tf.convert_to_tensor(prediction_bbox, dtype=tf.float32)
        prediction_type = tf.convert_to_tensor(prediction_type, dtype=tf.int64)
        prediction_score = tf.convert_to_tensor(prediction_score, dtype=tf.float32)
        prediction_overlap_nlz = tf.convert_to_tensor(prediction_overlap_nlz, dtype=tf.bool)
        ground_truth_frame_id = tf.convert_to_tensor(ground_truth_frame_id, dtype=tf.int64)
        ground_truth_bbox = tf.convert_to_tensor(ground_truth_bbox, dtype=tf.float32)
        ground_truth_type = tf.convert_to_tensor(ground_truth_type, dtype=tf.int64)
        ground_truth_difficulty = tf.convert_to_tensor(ground_truth_difficulty, dtype=tf.int64)
        ground_truth_speed = None
        recall_at_precision = None
        results = \
            detection_metrics.get_detection_metric_ops(
                config,
                prediction_frame_id,
                prediction_bbox,
                prediction_type,
                prediction_score,
                prediction_overlap_nlz,
                ground_truth_frame_id,
                ground_truth_bbox,
                ground_truth_type,
                ground_truth_difficulty,
                ground_truth_speed,
                recall_at_precision,
            )
        return results

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.waymo_infos) * self.total_epochs

        return len(self.waymo_infos)

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.waymo_infos)

        info = copy.deepcopy(self.waymo_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)
        input_dict = {
            'points': points,
            'frame_id': sample_idx,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = FrameParser.drop_laser_labels_with_name(laser_labels=annos, keeped_category_types=self.class_names)
            gt_names = np.asarray([label_pb2.Label.Type.Name(anno.type) for anno in annos]) 
            gt_boxes_lidar = np.asarray([[anno.box.center_x,anno.box.center_y,anno.box.center_z,anno.box.length,anno.box.width,anno.box.height,anno.box.heading] for anno in annos])
            gt_boxes_points_num= np.asarray([anno.num_lidar_points_in_box for anno in annos])

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar,
                'gt_boxes_points_num': gt_boxes_points_num
            })
            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_dataset(dataset_dir, output_dir, untar=True, parse_tfrecord=True, parse_label=True, parse_images=True, concate_image_feature_to_laser=True):
    import glob
    import tarfile
    import concurrent.futures as futures
    num_workers = 8

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1: unzip all files
    tar_files = glob.glob(str(dataset_dir /'source' / '*tar'))
    random.shuffle(tar_files)
    _output_dir = output_dir / 'untar'
    _output_dir.mkdir(parents=True, exist_ok=True)

    def single_process_untar(tar_file):
        print('untar : %s' % tar_file)
        file = tarfile.open(tar_file)
        file.extractall(_output_dir)  # specify which folder to extract to
        file.close()
    if untar:
        with futures.ThreadPoolExecutor(num_workers) as executor:
            executor.map(single_process_untar, tar_files)

    # 2: parse each .tfrecord file
    tfrecord_files = glob.glob(str(_output_dir / '*tfrecord'))
    random.shuffle(tfrecord_files)
    _output_dir = output_dir / 'sample'
    laser_output_dir = _output_dir / 'laser'
    point2pixel_output_dir = _output_dir / 'point2pixel'
    label_output_dir = _output_dir / 'label'
    laser_output_dir.mkdir(parents=True, exist_ok=True)
    point2pixel_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    if parse_images:
        image_output_dir = _output_dir / 'image'
        image_label_output_dir = _output_dir / 'image_label'
        image_output_dir.mkdir(parents=True, exist_ok=True)
        image_label_output_dir.mkdir(parents=True, exist_ok=True)

    def single_process_parse(tfrecord_file):
        frames = FrameParser.parse_tfrecord_to_frame(tfrecord_file)
        samples = []
        context_infos = {}
        print('parsing %s' % tfrecord_file)
        for index, frame in enumerate(frames):
            name = frame.context.name
            flag, points, point2pixel, laser_labels = FrameParser.parse_laser(frame, parse_label=parse_label)
            assert flag is True
            context_name = frame.context.name
            camera_calibrations = frame.context.camera_calibrations
            frame_timestamp_micros = frame.timestamp_micros

            if parse_images:
                camera_names = ['UNKNOWN', 'FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
                camera_images = {}
                camera_labels = {}
                for camera_name in camera_names:
                    success, image_array, camera_label = FrameParser.parse_image(frame, camera_name=camera_name, parse_label=parse_label)
                    if success:
                        camera_images[camera_name] = image_array
                        camera_labels[camera_name] = camera_label

            if parse_label:
                # label data
                laser_labels_pickle = []
                for laser_label in laser_labels:
                    laser_labels_pickle.append(laser_label)
                with open(label_output_dir / ('%s_%s.bin' % (name, index)), 'wb') as f:
                    pickle.dump(laser_labels_pickle, f)

            if parse_images:
                for camera_name, image in camera_images.items():
                    with open(image_output_dir / ('%s_%s_%s.bin' % (name, index, camera_name)), 'wb') as f:
                        pickle.dump(camera_images[camera_name], f)
                    with open(image_label_output_dir / ('%s_%s_%s.bin' % (name, index, camera_name)), 'wb') as f:
                        # pickle can not process Repeated message , convert to list
                        _camera_labels = [camera_label for camera_label in camera_labels[camera_name]]
                        pickle.dump(_camera_labels, f)
                if concate_image_feature_to_laser:
                    _point2pixel = point2pixel.reshape(-1, 2, 3).transpose((1,0,2))
                    camera_focus_info = {}
                    for camera_calibration in camera_calibrations:
                        # see CameraCalibration in
                        # https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/dataset.proto
                        camera_name = dataset_pb2.CameraName.Name.Name(camera_calibration.name)
                        camera_focus_info[camera_name] = {
                            'f_u': camera_calibration.intrinsic[0],
                            'f_v': camera_calibration.intrinsic[1]
                        }
                    features = FrameParser.extract_image_features_to_laser_points(
                        camera_images=camera_images, camera_labels=camera_labels, point2pixel=_point2pixel, points=points, camera_focus_info=camera_focus_info)
                    points = np.concatenate((points, features), axis=1)

            # laser data
            '''
            sum time in seconds of every 100 iteration
                tf.data.TFRecordDataset: 18
                pickle: 0.03765726089477539
                numpy.save/load: 0.05744194984436035
                we adopt pickle
                example of TFRecordDataset:
                        write:
                        with tf.io.TFRecordWriter(filename) as writer:
                            _proto = dataset_pb2.MatrixFloat(data=points_all.flatten(), shape=dataset_pb2.MatrixShape(dims=points_all.shape))
                            writer.write(_proto.SerializeToString())
                            writer.close()

                        read:
                        tfrecords = tf.data.TFRecordDataset(tfrecord_file, compression_type='')
                        for data in tfrecords:
                            frame = dataset_pb2.MatrixFloat()
                            frame.ParseFromString(bytearray(data.numpy()))
                            _data = np.reshape(np.array(frame.data), frame.shape.dims)

            '''
            with open(laser_output_dir / ('%s_%s.bin' % (name, index)), 'wb') as f:
                pickle.dump(points, f)
            with open(point2pixel_output_dir / ('%s_%s.bin' % (name, index)), 'wb') as f:
                pickle.dump(point2pixel, f)

            samples.append('%s_%s' % (name, index))
            context_infos['%s_%s' % (name, index)] = {
                'context_name': context_name,
                'frame_timestamp_micros': frame_timestamp_micros
            }
        return samples, context_infos

    # single_process_parse(tfrecord_file=tfrecord_files[0])  # debug
    if parse_tfrecord:
        with futures.ThreadPoolExecutor(num_workers) as executor:
            sample_sets = []
            context_infos = []
            pbar = tqdm.tqdm(total=len(tfrecord_files))
            for sample_set, context_info in executor.map(single_process_parse, tfrecord_files):
                sample_sets.append(sample_set)
                context_infos.append(context_info)
                pbar.update(1)
            pbar.close()
        _sample_sets = []
        for sample_set in sample_sets: _sample_sets += sample_set
        with open(_output_dir / 'dataset.bin', 'wb') as f:
            pickle.dump(_sample_sets, f)

        _context_infos = dict(ChainMap(*context_infos))
        with open(_output_dir / 'context_infos.bin', 'wb') as f:
            pickle.dump(_context_infos, f)


def create_waymo_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = WaymoDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False, split='test')
    train_split, val_split, test_split = 'training', 'validation', 'testing'

    train_filename = save_path / ('waymo_infos_%s.pkl' % train_split)
    val_filename = save_path / ('waymo_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'waymo_infos_trainval.pkl'
    test_filename = save_path / ('waymo_infos_%s.pkl' % test_split)

    data_split_path = Path(dataset_cfg.DATA_PATH) / train_split
    parse_dataset(dataset_dir=data_split_path,output_dir=data_split_path, untar=False, parse_tfrecord=True, parse_label=True, parse_images=True, concate_image_feature_to_laser=True)
    data_split_path = Path(dataset_cfg.DATA_PATH) / val_split
    parse_dataset(dataset_dir=data_split_path, output_dir=data_split_path, untar=True, parse_tfrecord=True, parse_label=True, parse_images=True, concate_image_feature_to_laser=True)
    data_split_path = Path(dataset_cfg.DATA_PATH) / test_split
    parse_dataset(dataset_dir=data_split_path, output_dir=data_split_path, untar=True, parse_tfrecord=True, parse_label=False, parse_images=True, concate_image_feature_to_laser=False)

    print('---------------Start to generate data infos---------------')

    dataset.set_split('train')
    waymo_infos_train = dataset.get_infos(num_workers=workers, has_label=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(waymo_infos_train, f)
    print('Waymo info train file is saved to %s' % train_filename)
        
    dataset.set_split('valid')
    waymo_infos_val = dataset.get_infos(num_workers=workers, has_label=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(waymo_infos_val, f)
    print('Waymo info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(waymo_infos_train + waymo_infos_val, f)
    print('Waymo info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    waymo_infos_test = dataset.get_infos(num_workers=workers, has_label=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(waymo_infos_test, f)
    print('Waymo info test file is saved to %s' % test_filename)
        
    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    # print('---------------Data preparation Done---------------')


def check_dataset(dataset_cfg, class_names, skip_index=0, training=True, split='train', root_path=None, logger=None):
    dataset = WaymoDataset(dataset_cfg, class_names, training=training, split=split, root_path=root_path, logger=logger)
    print('checking dataset')
    print(skip_index, len(dataset))
    for index in tqdm.tqdm(range(skip_index, len(dataset))):
        try:
            data = dataset.__getitem__(index)
        except Exception as e:
            print('Fail at {}'.format(index))
            return False
    return True


def debug_data(dataset_cfg, class_names, data_index=0, training=True, split='train', root_path=None, logger=None):
    dataset = WaymoDataset(dataset_cfg, class_names, training=training, split=split, root_path=root_path, logger=logger)
    print('debug data at {}'.format(data_index))
    data = dataset.__getitem__(data_index)


if __name__ == '__main__':
    dataset_dir = Path('data/waymo/training/source')
    output_dir = Path('data/waymo/training/output')
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_waymo_dataset':

        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()

        dataset_cfg = dataset_cfg
        class_names = ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST']
        data_path = ROOT_DIR / 'data' / 'waymo'
        save_path = ROOT_DIR / 'data' / 'waymo'
        # generate dataset
        create_waymo_infos(dataset_cfg=dataset_cfg,class_names=class_names,data_path=data_path,save_path=save_path)
    elif sys.argv.__len__() > 1 and sys.argv[1] == 'check_dataset':
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()

        dataset_cfg = dataset_cfg
        class_names = ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST']
        data_path = ROOT_DIR / 'data' / 'waymo'
        save_path = ROOT_DIR / 'data' / 'waymo'
        index = 0
        # checking dataset
        check_dataset(dataset_cfg, class_names, skip_index=index, training=True, split='train', root_path=data_path)
    elif sys.argv.__len__() > 1 and sys.argv[1] == 'debug_data':
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()

        dataset_cfg = dataset_cfg
        class_names = ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST']
        data_path = ROOT_DIR / 'data' / 'waymo'
        save_path = ROOT_DIR / 'data' / 'waymo'
        index = 0 
        # debug dataset
        debug_data(dataset_cfg, class_names, data_index=index, training=True, split='train', root_path=data_path)
