import numpy as np
from pyquaternion import Quaternion
from scipy.interpolate import make_interp_spline
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import PointCloud
from nuscenes.utils.geometry_utils import view_points
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap

def process_frame(nusc, sample, plans):
    camera_token = sample['data']['CAM_FRONT']
    camera_data = nusc.get('sample_data', camera_token)

    calibrated_sensor = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
    translation = calibrated_sensor['translation']  # [x, y, z]
    rotation = calibrated_sensor['rotation']        # Quaternion [w, x, y, z]

    camera_intrinsic = calibrated_sensor['camera_intrinsic']
    # print(np.array(camera_intrinsic))
    point_ego = np.array(plans)  # 6 x 3
    point_ego[:, 2] = 0
    point_ego[:, 0] -= .5
    point_ego = np.insert(point_ego, 3, 1, axis=1)
    translation = np.array(translation)
    rotation = Quaternion(rotation).rotation_matrix

    ego_to_cam = np.eye(4)
    ego_to_cam[:3, :3] = rotation
    ego_to_cam[:3, 3] = translation
    # print(ego_to_cam)
    ego_to_cam = np.linalg.inv(ego_to_cam)
    # print(ego_to_cam)
    point_cam = np.dot(ego_to_cam, point_ego.T)[:3, :] # 3 x 6
    # point_cam = np.array([0, 1, 10])
    # if point_cam.ndim == 1:
    #     point_cam = np.expand_dims(point_cam, axis=1)

    point_img = view_points(point_cam, np.array(camera_intrinsic), normalize=True).T
    # point_img = np.dot(camera_intrinsic, point_cam)

    # u, v = point_img[0]/point_img[2], point_img[1]/point_img[2]
    # print(f"Point in image pixel coordinates: ({u}, {v})")

    u = point_img[2:, 0]
    v = point_img[2:, 1]
    u = u[::-1]
    v = v[::-1]
    x_smooth = np.linspace(v.min(), v.max(), 300)
    spl = make_interp_spline(v, u, k=3)
    y_smooth = spl(x_smooth)

    img_path = camera_data['filename']
    img = cv2.imread(img_path)
    img = cv2.resize(img, (1600, 900))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # for ii, (u, v, _) in enumerate(point_img):
    #     if u > 0 and v > 0:
    #         Green = np.array((100, 255, 200))
    #         Blue = np.array((100, 200, 255))
    #         Color = Green * (1-ii*0.1) + Blue * ii * 0.1
    #         cv2.circle(img, (int(u), int(v)), 10, Color, -1)
    #         if ii > 1:
    #             cv2.line(img, (int(u), int(v)), (int(point_img[ii-1][0]), int(point_img[ii-1][1])), Color, 3)


    
    # cmap = plt.get_cmap('summer').reversed()
    colors = ["yellow", "green", "green", "red"]
    cmap = LinearSegmentedColormap.from_list("mycmap", colors)

    norm = plt.Normalize(vmin=x_smooth.min(), vmax=x_smooth.max())
    points = np.array([y_smooth, x_smooth]).T.reshape(-1, 1, 2)

    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(x_smooth) 
    lc.set_linewidth(5) 
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(img)

    ax.add_collection(lc)
    # plt.plot(y_smooth, x_smooth, color='#44FF44')
    # plt.scatter([u], [v], color='#44FF44', s=30, zorder=3)
    # plt.show()
    # ax.xlim(0, 1600)
    # ax.ylim(950, 0) 
    ax.axis('off')