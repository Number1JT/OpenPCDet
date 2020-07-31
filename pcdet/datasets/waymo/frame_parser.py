import os
import tensorflow as tf
import math
import numpy as np
import itertools
import cv2
import pickle
import copy
import torch
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
''' 

from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
'''
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


enum_LabelType = label_pb2.Label.Type


class FrameParser:
    def __init__(self,):
        pass

    @staticmethod
    def parse_tfrecord(tfrecord_file):
        tfrecords = tf.data.TFRecordDataset(tfrecord_file, compression_type='')
        return tfrecords

    @staticmethod
    def parse_tfrecord_to_frame(tfrecord_file):
        tfrecord = FrameParser.parse_tfrecord(tfrecord_file)
        for data in tfrecord:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            frame.images.sort(key=lambda image: image.name)
            frame.camera_labels.sort(key=lambda label: label.name)
            frame.lasers.sort(key=lambda laser: laser.name)
            yield frame

    # @staticmethod
    # def objects2label_format(input_file, output_dir):
    #     '''
    #     transfer metrics_pb2.Objects to dataset_pb2.Label format
    #     :return:
    #     '''
    #     streams = open(input_file, 'rb').read()
    #     objects = metrics_pb2.Objects()
    #     objects.ParseFromString(bytearray(streams))
    #
    #     frame_labels_map = {}
    #     for object in objects.objects:
    #         context_name = object.context_name
    #         frame_timestamp_micros = object.frame_timestamp_micros
    #         label = object.object
    #         score = object.score
    #         camera_name = object.camera_name
    #
    #         if context_name not in frame_labels_map:
    #             frame_labels_map[context_name] = {}
    #         frame_labels_map[context_name][frame_timestamp_micros] = label
    #
    #     for frame, labels in frame_labels_map.items():
    #         _keys = frame_labels_map[frame].sort()
    #         for index, key in enumerate(_keys):
    #             frame_id = frame + '_%s' % index
    #             with open(Path(output_dir) / frame_id + '.bin', 'wb') as f:
    #                 pickle.dump(frame_labels_map[frame][key],f)

    @staticmethod
    def check_numpy_to_torch(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float(), True
        return x, False

    @staticmethod
    def rotate_points_along_z(points, angle):
        """
        Args:
            points: (B, N, 3 + C)
            angle: (B), angle along z-axis, angle increases x ==> y
        Returns:

        """
        points, is_numpy = FrameParser.check_numpy_to_torch(points)
        angle, _ = FrameParser.check_numpy_to_torch(angle)

        cosa = torch.cos(angle)
        sina = torch.sin(angle)
        zeros = angle.new_zeros(points.shape[0])
        ones = angle.new_ones(points.shape[0])
        rot_matrix = torch.stack((
            cosa, sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(-1, 3, 3).float()
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
        return points_rot.numpy() if is_numpy else points_rot

    @staticmethod
    def boxes_to_corners_3d(boxes3d):
        """
            7 -------- 4
           /|         /|
          6 -------- 5 .
          | |        | |
          . 3 -------- 0
          |/         |/
          2 -------- 1
        Args:
            boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

        Returns:
        """
        boxes3d, is_numpy = FrameParser.check_numpy_to_torch(boxes3d)
        template = boxes3d.new_tensor((
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        )) / 2

        corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
        corners3d = FrameParser.rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
        corners3d += boxes3d[:, None, 0:3]

        return corners3d.numpy() if is_numpy else corners3d

    @staticmethod
    def show_image_with_box(image, boxes, categorys, output_dir, name=None, thickness=1, font_scale=0.3):
        image = copy.deepcopy(image)
        image = np.ascontiguousarray(image[:, :, ::-1])  # RGB -> BGR
        boxes = copy.deepcopy(boxes)
        color = (0, 255, 255)
        for i in range(len(boxes)):
            # extract the bounding box coordinates
            cx, cy = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            xmin, ymin = cx - w / 2, cy - h / 2
            xmin, ymin, w, h = map(int, [xmin, ymin, w, h])
            category_index = categorys[i]
            category = enum_LabelType.Name(category_index)

            # draw a bounding box rectangle and label on the image
            cv2.rectangle(image, (xmin, ymin), (xmin + w, ymin + h), color=color, thickness=thickness)
            confidence = 1.0
            text = f"{category}: {confidence:.2f}"

            # calculate text width & height to draw the transparent boxes as background of the text
            (text_width, text_height) = \
                cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
            text_offset_x = xmin
            text_offset_y = ymin - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))

            overlay = image.copy()
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
            # add opacity (transparency to the box)
            image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
            # now put the text (label: confidence %)
            cv2.putText(image, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

        cv2.imwrite(os.path.join(output_dir, "image_detection_result_{}.jpg".format(name)), image)

    @staticmethod
    def show_laser_with_box(
            pcds, pcds_color=None,
            box3ds=None, box3d_color=[255, 0, 0],
            categorys=None,
            pred_box3ds=None, pred_box3d_color=[0, 0, 255],
            pred_categorys=None,
            x_range=(-40, 62.4), y_range=(-40, 40), z_range=(-3, 8),
            name='laser'
    ):
        '''

        :param pcds: pcds numpy float32 array (n, c)
        :param pcds_color: pcds_color uint8 array (n, 3) denotes the color of each point 0~255
        :param box3ds: box3d_corners float32 numpy array; eight corners of a 3D bounding box, [x,y,z,l,w,h,heading]
        :param box3d_color:
        :param categorys:
        :param pred_box3ds:
        :param pred_box3d_color:
        :param pred_categorys:
        :param x_range:
        :param y_range:
        :param z_range:
        :param name:
        :return:
        '''
        import open3d as o3d
        #
        #
        #
        pcd_l = o3d.geometry.PointCloud()
        pcd_l.points = o3d.utility.Vector3dVector(pcds[:, :3])
        if pcds_color is not None:
            pcd_l.colors = o3d.utility.Vector3dVector(pcds_color)

        lines = [[0, 1], [1, 5], [5, 4], [4, 0],
                 [3, 2], [2, 6], [6, 7], [7, 3],
                 [0, 3], [4, 7], [1, 2], [5, 6]]

        line_sets = []
        if box3ds is not None:
            box3d_corners = FrameParser.boxes_to_corners_3d(box3ds)
            for box3d_corner in box3d_corners:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(box3d_corner)

                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector([box3d_color for i in range(len(lines))])

                line_sets.append(line_set)

        if pred_box3ds is not None:
            pred_box3d_corners = FrameParser.boxes_to_corners_3d(pred_box3ds)
            for box3d_corner in pred_box3d_corners:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(box3d_corner)

                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector([pred_box3d_color for i in range(len(lines))])

                line_sets.append(line_set)

        line_set = o3d.geometry.LineSet()
        corners = []
        for z in z_range:
            corners.append([x_range[0], y_range[0], z])
            corners.append([x_range[0], y_range[1], z])
            corners.append([x_range[1], y_range[1], z])
            corners.append([x_range[1], y_range[0], z])

        line_set.points = o3d.utility.Vector3dVector(corners)
        colors = [[0, 0, 0] for i in range(len(lines))]
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_sets.append(line_set)
        o3d.visualization.draw_geometries([*line_sets, pcd_l], window_name=name)

    @staticmethod
    def convert_range_image_to_point_cloud(frame,
                                           range_images,
                                           camera_projections,
                                           range_image_top_pose,
                                           ri_index=0):
        """Convert range images to point cloud.
        clone from waymo_open_dataset(https://github.com/waymo-research/waymo-open-dataset)
        Args:
          frame: open dataset frame
           range_images: A dict of {laser_name, [range_image_first_return,
             range_image_second_return]}.
           camera_projections: A dict of {laser_name,
             [camera_projection_from_first_return,
             camera_projection_from_second_return]}.
          range_image_top_pose: range image pixel pose for top lidar.
          ri_index: 0 for the first return, 1 for the second return.

        Returns:
          points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars). [x, y, z]
          cp_points: {[N, 6]} list of camera projections of length 5
            (number of lidars).
        """
        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        points = []
        cp_points = []

        frame_pose = tf.convert_to_tensor(
            value=np.reshape(np.array(frame.pose.transform), [4, 4]))
        # [H, W, 6]
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image_top_pose.data),
            range_image_top_pose.shape.dims)
        # [H, W, 3, 3]
        range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
            range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
            range_image_top_pose_tensor[..., 2])
        range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation,
            range_image_top_pose_tensor_translation)
        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0
            range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian,
                                         tf.compat.v1.where(range_image_mask))

            cp = camera_projections[c.name][ri_index]
            cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor,
                                            tf.compat.v1.where(range_image_mask))
            points.append(points_tensor.numpy())
            cp_points.append(cp_points_tensor.numpy())

        return points, cp_points

    @staticmethod
    def convert_range_image_to_point_cloud_V2(frame,
                                           range_images,
                                           camera_projections,
                                           range_image_top_pose,
                                           ri_index=0):
        """Convert range images to point cloud.

        Args:
          frame: open dataset frame
           range_images: A dict of {laser_name, [range_image_first_return,
             range_image_second_return]}.
           camera_projections: A dict of {laser_name,
             [camera_projection_from_first_return,
             camera_projection_from_second_return]}.
          range_image_top_pose: range image pixel pose for top lidar.
          ri_index: 0 for the first return, 1 for the second return.

        Returns:
          points: {[N, 4]} list of 3d lidar points of length 5 (number of lidars). [x, y, z, intensity, elongation]
          cp_points: {[N, 6]} list of camera projections of length 5
            (number of lidars).
        """
        calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
        points = []
        cp_points = []

        frame_pose = tf.convert_to_tensor(
            value=np.reshape(np.array(frame.pose.transform), [4, 4]))
        # [H, W, 6]
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image_top_pose.data),
            range_image_top_pose.shape.dims)
        # [H, W, 3, 3]
        range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
            range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
            range_image_top_pose_tensor[..., 2])
        range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation,
            range_image_top_pose_tensor_translation)
        for c in calibrations:
            range_image = range_images[c.name][ri_index]
            if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0
            range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)
            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            # see waymo_open_dataset/dataset.proto RangeImage.range_image_compressed
            range_image_intensity = tf.expand_dims(range_image_tensor[..., 1], axis=-1)
            range_image_elongation = tf.expand_dims(range_image_tensor[..., 2], axis=-1)
            range_image_features = tf.concat((range_image_cartesian, range_image_intensity, range_image_elongation), axis=-1)
            points_tensor = tf.gather_nd(range_image_features,
                                         tf.compat.v1.where(range_image_mask))

            cp = camera_projections[c.name][ri_index]
            cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor,
                                            tf.compat.v1.where(range_image_mask))
            points.append(points_tensor.numpy())
            cp_points.append(cp_points_tensor.numpy())

        return points, cp_points

    @staticmethod
    def parse_image(frame, camera_name='UNKNOWN', parse_label=True, output_dir='', vis=False):
        assert camera_name in dataset_pb2.CameraName.Name.keys()
        camera_name_index = dataset_pb2.CameraName.Name.Value(camera_name)
        image_array = None
        for image in frame.images:
            if image.name == camera_name_index:
                image_tensor = tf.image.decode_jpeg(image.image)
                image_array = image_tensor.numpy()
                break
        camera_labels = None
        if parse_label:
            for camera_labels in frame.camera_labels:
                if camera_labels.name == camera_name_index:
                    camera_labels = camera_labels.labels
                    if vis:
                        boxes, categorys = [], []
                        for label in camera_labels:
                            box, category = label.box, label.type
                            # see https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto
                            center_x, center_y, w, h = box.center_x, box.center_y, box.length, box.width
                            boxes.append([center_x, center_y, w, h])
                            categorys.append(category)
                        boxes = np.asarray(boxes)
                        categorys = np.asarray(categorys)
                        FrameParser.show_image_with_box(image_array, boxes, categorys, output_dir, camera_name)
                    break
        if image_array is None:
            success = False
        else:
            success = True
        return success, image_array, camera_labels

    @staticmethod
    def drop_laser_labels_with_name(laser_labels, keeped_category_types):
        '''
        only keep laser label which category in category_names
        :param laser_labels: iterative label_pb2.Label
        :param keeped_category_types: list string
        :return: iterative label_pb2.Label
        '''
        _laser_labels = []
        for laser_label in laser_labels:
            _type = label_pb2.Label.Type.Name(laser_label.type)
            if _type in keeped_category_types:
                _laser_labels.append(laser_label)
        return _laser_labels

    @staticmethod
    def parse_laser_labels(laser_labels):
        laser_box3d, laser_box3d_categorys, detection_difficulty_level, num_lidar_points_in_box = [], [], [], []
        for laser_label in laser_labels:
            box = laser_label.box
            center_x, center_y, center_z = box.center_x, box.center_y, box.center_z
            width, length, height = box.width, box.length, box.height
            heading = box.heading
            category = enum_LabelType.Name(laser_label.type)
            # see https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto
            laser_box3d.append([center_x, center_y, center_z, length, width, height, heading])
            laser_box3d_categorys.append(category)
            detection_difficulty_level.append(laser_label.detection_difficulty_level)
            num_lidar_points_in_box.append(laser_label.num_lidar_points_in_box)
        laser_box3d = np.asarray(laser_box3d)
        laser_box3d_categorys = np.asarray(laser_box3d_categorys)
        detection_difficulty_level = np.asarray(detection_difficulty_level)
        num_lidar_points_in_box = np.asarray(num_lidar_points_in_box)
        return laser_box3d, laser_box3d_categorys, detection_difficulty_level, num_lidar_points_in_box

    @staticmethod
    def parse_laser(frame, parse_label=True, output_dir="", vis=False):
        '''

        :param parse_label:
        :param vis:
        :return:
            points_all: np.ndarray
                [N, 4] list of 3d lidar points of length 3 or 5.
                [x, y, z, intensity(optional), elongation(optional)]
            laser labels: format https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto
                message Label {
                  // Upright box, zero pitch and roll.
                  message Box {
                    optional double center_x = 1;
                    optional double center_y = 2;
                    optional double center_z = 3;
                    optional double length = 5;
                    optional double width = 4;
                    optional double height = 6;
                    optional double heading = 7;
                  }
                  optional Box box = 1;

                  enum Type {
                    TYPE_UNKNOWN = 0;
                    TYPE_VEHICLE = 1;
                    TYPE_PEDESTRIAN = 2;
                    TYPE_SIGN = 3;
                    TYPE_CYCLIST = 4;
                  }
                  optional Type type = 3;

                  enum DifficultyLevel {
                    UNKNOWN = 0;
                    LEVEL_1 = 1;
                    LEVEL_2 = 2;
                  }
                  optional DifficultyLevel detection_difficulty_level = 5;
                  optional DifficultyLevel tracking_difficulty_level = 6;
                  optional int32 num_lidar_points_in_box = 7;
                }
        '''
        range_images, camera_projections, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
        points, cp_points = FrameParser.convert_range_image_to_point_cloud_V2(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=0
        )

        # 3d points in vehicle frame.
        points_all = np.concatenate(points, axis=0)
        # camera projection corresponding to each point.
        cp_points_all = np.concatenate(cp_points, axis=0)
        laser_labels = frame.laser_labels
        if parse_label:
            if vis:
                laser_box3d, laser_box3d_categorys, detection_difficulty_level, num_lidar_points_in_box \
                    = FrameParser.parse_laser_labels(laser_labels)
                np.save(os.path.join(output_dir, "laser_points.npy"), points_all)
                np.save(os.path.join(output_dir, "laser_cp_points.npy"), cp_points_all)
                np.save(os.path.join(output_dir, "laser_box3d.npy"), laser_box3d)
                np.save(os.path.join(output_dir, "laser_box3d_categorys.npy"), laser_box3d_categorys)
                # FrameParser.show_laser_with_box(pcds=points_all, box3ds=boxes, categorys=categorys)
        return True, points_all, cp_points_all, laser_labels

    @staticmethod
    def extract_image_features_to_laser_points(camera_images, camera_labels, point2pixel, points=None, camera_focus_info=None):
        """

        :param camera_images: dict, key:camera_name value:pixel_array
        :param camera_labels: dict, key:camera_name value:box label in label_pb2.Label format
        :param point2pixel: np.array , point to pixel map in RangeImage.camera_projection_compressed format ,shape N*6,
        :return:
        """
        # Lidar point to camera image projections. A point can be projected to
        # multiple camera images. We pick the first two at the following order:
        # [FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT].
        multi_camera, points_num, dims = point2pixel.shape
        point2pixel = point2pixel.reshape((points_num * multi_camera , dims))  # index, loc_x, loc_y
        if points is not None:
            assert points.shape[0] == points_num
            multi_points = points.reshape(1, points_num, -1).\
                repeat(multi_camera, axis=0).\
                reshape(multi_camera * points_num, -1)
        semantic_features = np.zeros((points_num * multi_camera, 1), dtype=np.float)  # category
        geometric_features = np.zeros((points_num * multi_camera, 5), dtype=np.float) # see ImVoteNet paper
        texture_features = np.zeros((points_num * multi_camera, 3), dtype=np.float)  # RGB

        # convert camera_labels to np.array format
        camera_labels_array = {}
        for camera_name, camera_label in camera_labels.items():
            camera_labels_array[camera_name] = np.asarray(
                [[_label.box.center_x, _label.box.center_y, _label.box.length, _label.box.width, _label.type] for _label in camera_label]
            )

        camera_names = camera_images.keys()
        for camera_name in camera_names:
            camera_label = camera_labels_array[camera_name]
            np.random.shuffle(camera_label)  # shuttle array in axis 0
            camera_index = dataset_pb2.CameraName.Name.Value(camera_name)
            camera_image = camera_images[camera_name]
            # if camera_index == 0:  # not projection to any image
            #     continue

            index = point2pixel[:, 0] == camera_index
            if index.any() is False:
                continue
            N = index.sum()
            pixel_loc_x = point2pixel[index, 1]
            pixel_loc_y = point2pixel[index, 2]

            # extract texture features
            texture_features[index] = camera_image[pixel_loc_y, pixel_loc_x]

            # extract semantic features
            # extract geometric features
            if camera_label.size == 0:
                semantic_features[index, 0] = -1  # background
                # geometric_features[index] = 0
            else:
                top_left_x = camera_label[:, 0] - camera_label[:, 2] / 2
                top_left_y = camera_label[:, 1] - camera_label[:, 3] / 2
                bottom_right_x = camera_label[:, 0] + camera_label[:, 2] / 2
                bottom_right_y = camera_label[:, 1] + camera_label[:, 3] / 2
                # N * M, N points num, M boxes num
                match = (pixel_loc_x[:, None] > top_left_x[None, :]) \
                          & (pixel_loc_x[:, None] < bottom_right_x[None, :]) \
                          & (pixel_loc_y[:, None] > top_left_y[None, :]) \
                          & (pixel_loc_y[:, None] < bottom_right_y[None, :])
                # assert match.sum(axis=1) <= 1  # point may matched to multi box
                label_feature = np.zeros(N, np.float)
                geometric_feature = np.zeros((N, 5), np.float)
                bg = match.any(axis=1) == False
                label_feature[bg] = -1  # background
                geometric_feature[bg] = 0

                fg = match.any(axis=1) == True
                match_fg = match[fg]
                box_argmax = match_fg.argmax(axis=1)  # find first matched box
                matched_camera_label = camera_label[box_argmax, -1]  # -1 is the label of box
                matched_camera_center_x = camera_label[box_argmax, 0]
                matched_camera_center_y = camera_label[box_argmax, 1]

                label_feature[fg] = matched_camera_label
                semantic_features[index, 0] = label_feature

                delta_u = pixel_loc_x[fg] - matched_camera_center_x
                delta_v = pixel_loc_y[fg] - matched_camera_center_y
                assert points is not None
                assert camera_focus_info is not None
                assert points.shape[0] == points_num
                f_u, f_v = camera_focus_info[camera_name]['f_u'], \
                           camera_focus_info[camera_name]['f_v']
                point_x = multi_points[index][fg, 0]
                point_y = multi_points[index][fg, 1]
                point_z = multi_points[index][fg, 2]
                # NOTE that u,v in Image correspond to y,z in Laser coordinate
                depth = point_x
                oc_ = np.stack(
                    (point_y + delta_u / f_u * depth, point_z + delta_v / f_v * depth, depth), axis=-1)
                oc_len = np.linalg.norm(oc_, axis=1, keepdims=True)
                oc_normalize = oc_ / oc_len
                geometric_feature[fg, 0] = delta_u / f_u * depth
                geometric_feature[fg, 1] = delta_v / f_v * depth
                geometric_feature[fg, 2:5] = oc_normalize
                geometric_features[index] = geometric_feature

        # if laser point matched to multi camera image, only keep one matched feature
        point2pixel = point2pixel.reshape(multi_camera, points_num, -1)
        semantic_features = semantic_features.reshape(multi_camera, points_num, -1)
        texture_features = texture_features.reshape(multi_camera, points_num, -1)
        geometric_features = geometric_features.reshape(multi_camera, points_num, -1)
        _semantic_features = np.zeros(shape=(points_num, 1), dtype=np.float)
        _geometric_features = np.zeros(shape=(points_num, 5), dtype=np.float)
        _texture_features = np.zeros(shape=(points_num, 3), dtype=np.float)  # RGB
        for i in range(multi_camera):
            _point2pixel = point2pixel[i]
            fg = _point2pixel[:, 0] != 0
            _semantic_features[fg] = semantic_features[i][fg]
            _texture_features[fg] = texture_features[i][fg]
            _geometric_features[fg] = geometric_features[i][fg]
        _texture_features = _texture_features / 256  # normalize to [0,1]
        return np.concatenate((_semantic_features, _texture_features, _geometric_features), axis=-1)


if __name__ == '__main__':
    _dir = '/data/'
    tfrecord_file = os.path.join(_dir, 'frames')
    for frame in FrameParser.parse_tfrecord_to_frame(tfrecord_file):
        FrameParser.parse_image(frame, 'FRONT', vis=True)
        FrameParser.parse_image(frame, 'FRONT_LEFT', vis=True)
        FrameParser.parse_image(frame, 'FRONT_RIGHT', vis=True)
        FrameParser.parse_image(frame, 'SIDE_LEFT', vis=True)
        FrameParser.parse_image(frame, 'SIDE_RIGHT', vis=True)
        FrameParser.parse_laser(frame, parse_label=True, vis=True)

        laser_points = np.load(os.path.join(_dir,'laser_points.npy'))
        laser_box3d = np.load(os.path.join(_dir,'laser_box3d.npy'))
        laser_box3d_categorys = np.load(os.path.join(_dir,'laser_box3d_categorys.npy'))
        FrameParser.show_laser_with_box(pcds=laser_points, box3ds=laser_box3d, categorys=laser_box3d_categorys)
