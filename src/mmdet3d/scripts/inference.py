#!/usr/bin/env python
# encoding: utf-8
from mmdet3d.apis import init_model, inference_detector,inference_multi_modality_detector
import numpy  as np
import torch
from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)

from geometry_msgs.msg import Point

import roslib; roslib.load_manifest('visualization_marker_tutorials')
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import rospy

from sensor_msgs.msg import PointCloud2

from sensor_msgs.msg import Image

from pypcd import pypcd
from rospy.numpy_msg import numpy_msg
# 多话题同步订阅
import message_filters as mf


# config_file =  'src/mmdet3d/scripts/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
# checkpoint_file = 'src/mmdet3d/scripts/checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth'
# 配置文件和模型文件
config_file = 'src/mmdet3d/scripts/configs/pointpillars/pointpillars_with_img.py'
checkpoint_file = 'src/mmdet3d/scripts/checkpoints/epoch_160.pth'


LINES = [[0, 1], [1, 2], [2, 3], [3, 0]] # lower face
LINES+= [[4, 5], [5, 6], [6, 7], [7, 4]] # upper face
LINES+= [[4, 0], [5, 1], [6, 2], [7, 3]] # connect lower face and upper face
# LINES+= [[4, 1], [5, 0]] # front face
# LINES+= [[4, 1], [0, 5]] # front face

Fram_id = ''
RATE = 1
LIFETIME = 1.0/RATE

IMAGE_WIDTH = 1920
IMAGE_LENGTH = 1200

def convert_valid_bboxes(box_dict):
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (torch.Tensor): Scores of boxes.
                - labels_3d (torch.Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.

                - bbox (np.ndarray): 2D bounding boxes.
                - box3d_camera (np.ndarray): 3D bounding boxes in \
                    camera coordinate.
                - box3d_lidar (np.ndarray): 3D bounding boxes in \
                    LiDAR coordinate.
                - scores (np.ndarray): Scores of boxes.
                - label_preds (np.ndarray): Class label predictions.
                - sample_idx (int): Sample index.
        """
        rect = [1.3644980834165910e+03, 0., 1.0072530695210310e+03, 
                0., 1.3700616787937638e+03, 5.3653660361220579e+02, 
                0., 0., 1.]

        trv2c = [6.927964000000e-03, -9.999722000000e-01, -2.757829000000e-03, -2.457729000000e-02,
                        -1.162982000000e-03, 2.749836000000e-03, -9.999955000000e-01,-6.127237000000e-02, 
                        9.999753000000e-01, 6.931141000000e-03, -1.143899000000e-03, -3.321029000000e-01]

        rect = np.reshape(rect,(3,3)).astype(np.float32)
        Trv2c = np.reshape(trv2c,(3,4)).astype(np.float32)
        img_shape = [1200,1920]
        P2 = np.eye(4)  

        pcd_limit_range=[-70.4, -40, -3, 70.4, 40, 0.0]
        # TODO: refactor this function
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        # sample_idx = info['image']['image_idx']
        # TODO: remove the hack of yaw
        box_preds.tensor[:, -1] = box_preds.tensor[:, -1] - np.pi
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]))
                # sample_idx=sample_idx)
     
        # rect = info['calib']['R0_rect'].astype(np.float32)
        # Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        # P2 = info['calib']['P2'].astype(np.float32)
        # img_shape = info['image']['image_shape']
        # P2 = box_preds.tensor.new_tensor(P2)
        # Trv2c = np.reshape(trv2c,(3,4)).astype(np.float32)
        P2 = box_preds.tensor.new_tensor(P2)
        box_preds_camera = box_preds.convert_to(Box3DMode.CAM, rect @ Trv2c)

        box_corners = box_preds_camera.corners
        # print(box_corners)
        box_corners_in_image = points_cam2img(box_corners, P2)
        # box_corners_in_image: [N, 8, 2]
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)
        # Post-processing
        # check box_preds_camera
        image_shape = box_preds.tensor.new_tensor(img_shape)
        valid_cam_inds = ((box_2d_preds[:, 0] < image_shape[1]) &
                          (box_2d_preds[:, 1] < image_shape[0]) &
                          (box_2d_preds[:, 2] > 0) & (box_2d_preds[:, 3] > 0))
        # check box_preds
        limit_range = box_preds.tensor.new_tensor(pcd_limit_range)
        valid_pcd_inds = ((box_preds.center > limit_range[:3]) &
                          (box_preds.center < limit_range[3:]))
        valid_inds =  valid_pcd_inds.all(-1)

        if valid_inds.sum() > 0:
            return dict(
                bbox=box_2d_preds[valid_inds, :].numpy(),
                box3d_camera=box_preds_camera[valid_inds].tensor.numpy(),
                # box3d_lidar=box_preds[valid_inds].tensor.numpy(),
                box3d_lidar=box_preds.tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),)
                # sample_idx=sample_idx)
        else:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]))
                # sample_idx=sample_idx)

# 计算每个3d框的8个顶点
def compute_3d_box(bbox):
    """
    :param bbox:
    bbox include (x,y,z,l,w,h,yaw)
    :return:
    3xn in velo coordinate
    """
    yaw = bbox[6]
    w = bbox[4]
    l = bbox[3]
    h = bbox[5]
    x = bbox[0]
    y = bbox[1]
    z = bbox[2]
    R = np.array([[np.cos(yaw), np.sin(yaw), 0], [-np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    z_corners = [-h/2,-h/2,-h/2,-h/2,h/2,h/2,h/2,h/2]

    corners_3d_velo = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d_velo += np.vstack([x, y, z+1])
    return corners_3d_velo
    

# 发布3d框
def publish_3dbbox(box3d_pub,corners_3d):
    marker_array = MarkerArray()
    for i , corners_3d_velo in enumerate(corners_3d):
        marker = Marker()
        marker.header.frame_id = Fram_id
        marker.header.stamp = rospy.Time.now()

        marker.id = i
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(0.2)
        marker.type = Marker.LINE_LIST

        b, g, r = (255,255,0)
        marker.color.r = r/255.0
        marker.color.g = g/255.0
        marker.color.b = b/255.0

        marker.color.a = 1.0

        marker.scale.x = 0.1
        marker.points = []
        for l in LINES:
            p1 = corners_3d_velo[l[0]]
            marker.points.append(Point(p1[0], p1[1], p1[2]))
            p2 = corners_3d_velo[l[1]]
            marker.points.append(Point(p2[0], p2[1], p2[2]))
        marker_array.markers.append(marker)
    box3d_pub.publish(marker_array)


# 发布框的信息，跟踪算法订阅
def pub_bbox_info(bbox_info_puber, bboxes):
    # print(len(bboxes))
    marker_array_info = MarkerArray()
    for i in range(len(bboxes)):
        marker = Marker()
        # frame_id改为与传感器点云一致
        marker.header.frame_id = Fram_id
        marker.ns = 'detection'
        marker.type = marker.CUBE
        marker.action = marker.ADD

        marker.scale.x = bboxes[i][3]
        marker.scale.y = bboxes[i][4]
        marker.scale.z = bboxes[i][5]

        marker.pose.orientation.x = bboxes[i][6]
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 0.0
        marker.pose.position.x = bboxes[i][0]
        marker.pose.position.y = bboxes[i][1]
        marker.pose.position.z = bboxes[i][2]
        marker.color.a = 1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.id = i
        marker.lifetime = rospy.Duration(1)

        marker_array_info.markers.append(marker)
    bbox_info_puber.publish(marker_array_info)

# 单一点云检测的回调函数
def detecCallBack(pcd2_data):
    # 从topic读数据
    pc = pypcd.PointCloud.from_msg(pcd2_data)

    x = pc.pc_data['x'].flatten()
    y = pc.pc_data['y'].flatten()
    z = pc.pc_data['z'].flatten()
    intensity = pc.pc_data['intensity'].flatten()
    # 将topic形式的数据转换成模型需要的numpy数据格式
    point_data = np.zeros(x.shape[0] + y.shape[0] + z.shape[0] + intensity.shape[0], dtype=np.float32)

    point_data[::4] = x
    point_data[1::4] = y
    point_data[2::4] = z
    point_data[3::4] = 0


    point_data = point_data.astype(float)

    result, data = inference_detector(model, point_data)
    bbox = convert_valid_bboxes(result[0])['box3d_lidar']

    corners_3d_velos = []
    de_ids = []

    print(result[0]['labels_3d'])
    # 条件过滤，car的置信度设高，其余低
    ziped = zip(result[0]['scores_3d'],result[0]['labels_3d'])
    for id,(score,label) in enumerate(ziped):
        # if score <0.7 :
        #     de_ids.append(id)
        if label != 1:
            de_ids.append(id)

    bboxes  = np.delete(bbox, de_ids,axis=0)
    # 将各个框的表达方式转换成8个顶点
    for box in bboxes:
        corners_3d_velo = compute_3d_box(box).T
        corners_3d_velos += [corners_3d_velo]
    # 发布3d框
    publish_3dbbox(bbox_publisher,corners_3d_velos)
    # pub_bbox_info(bbox_info_puber, bboxes)


# 融合算法的回调函数
def multicallback(sub_pcd,sub_img):
    # 从topic读数据
    pc = pypcd.PointCloud.from_msg(sub_pcd)
    x = pc.pc_data['x'].flatten()
    y = pc.pc_data['y'].flatten()
    z = pc.pc_data['z'].flatten()
    intensity = pc.pc_data['intensity'].flatten()

    point_data = np.zeros(x.shape[0] + y.shape[0] + z.shape[0] + intensity.shape[0], dtype=np.float32)
    # 将topic形式的数据转换成模型需要的numpy数据格式
    point_data[::4] = x
    point_data[1::4] = y
    point_data[2::4] = z
    point_data[3::4] = 0
    point_data = point_data.astype(float)

    input_img = sub_img
    sub_img = np.frombuffer(input_img.data,dtype= np.uint8).reshape(input_img.height,input_img.width,-1)

    result, data = inference_multi_modality_detector(model, point_data,sub_img)
    bbox = convert_valid_bboxes(result[0])['box3d_lidar']

    corners_3d_velos = []
    de_ids = []

    ziped = zip(result[0]['scores_3d'], result[0]['labels_3d'])
    # 条件过滤，car的置信度设高，其余低
    for id, (score, label) in enumerate(ziped):
        if label == 2:
            if score < 0.7:
                de_ids.append(id)
        else:
            if score < 0.3:
                de_ids.append(id)

    bboxes = np.delete(bbox,de_ids,axis=0)
    # 将各个框的表达方式转换成8个顶点
    for box in bboxes:
        corners_3d_velo = compute_3d_box(box).T
        corners_3d_velos += [corners_3d_velo]

    # 发布3d框
    publish_3dbbox(bbox_publisher,corners_3d_velos)
    # 发布3d框给tracking
    # pub_bbox_info(bbox_info_puber, bboxes)


# 单点云检测
def pcd_listener():
    # 订阅点云topic,topic名称可改
    rospy.Subscriber('/rslidar_points',PointCloud2,detecCallBack)
    rospy.spin()


# 点云与图像融合检测
def pcd_and_img_listener():
   # 订阅点云topic,名称改为ros发布的pointcloud2名称
   sub_pcd = mf.Subscriber('/rslidar_points', PointCloud2)
   # 订阅图像topic,名称改为ros发布的image_raw名称
   sub_img = mf.Subscriber('/usb_cam/image_raw', numpy_msg(Image))
   # 同步订阅点云与图像
   sync = mf.ApproximateTimeSynchronizer([sub_pcd, sub_img],10,1)
   sync.registerCallback(multicallback)
   rospy.spin()

    
if __name__ == '__main__':
    # 检测模块节点名称
    rospy.init_node('detection',anonymous=True)
    model = init_model(config_file, checkpoint_file, device='cuda:0')
    markerArray = MarkerArray()
    # 发布的3d框 topic
    topic = '/detection/lidar_detector/objectsbbox'
    # 发布给tracking的框topic
    info_topic = '/bbox_info'
    bbox_publisher = rospy.Publisher(topic, MarkerArray, queue_size=1)
    bbox_info_puber = rospy.Publisher(info_topic, MarkerArray, queue_size=1)
    # 使用点云检测
    # pcd_listener()
    # 使用融合检测
    pcd_and_img_listener()
