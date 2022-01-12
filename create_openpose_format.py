import ipdb
import sys
import os.path as osp

import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
from glob import glob
import cv2
import mmcv
import pathlib
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Preprocess Panoptic')
parser.add_argument('panoptic_path')
parser.add_argument('dataset_name')
parser.add_argument('sequence_idx')
parser.add_argument('output_path')


def projectPoints(X, K, R, t, Kd):
    """ Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].

    Roughly, x = K*(R*X + t) + distortion

    See http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
    or cv2.projectPoints
    """

    x = np.asarray(R @ X + t)

    x[0:2, :] = x[0:2, :] / x[2, :]

    r = x[0, :] * x[0, :] + x[1, :] * x[1, :]

    x[0, :] = x[0, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r) + 2 * Kd[2] * x[0, :] * x[1, :] + Kd[3] * (
            r + 2 * x[0, :] * x[0, :])
    x[1, :] = x[1, :] * (1 + Kd[0] * r + Kd[1] * r * r + Kd[4] * r * r * r) + 2 * Kd[3] * x[0, :] * x[1, :] + Kd[2] * (
            r + 2 * x[1, :] * x[1, :])

    x[0, :] = K[0, 0] * x[0, :] + K[0, 1] * x[1, :] + K[0, 2]
    x[1, :] = K[1, 0] * x[0, :] + K[1, 1] * x[1, :] + K[1, 2]

    return x


EXPAND_FACTOR = 0.1

scale = (832, 512)
kpts_coco19 = [1,0,9,10,11,3,4,5,2,12,13,14,6,7,8,17,15,18,16]

def process_panoptic(panoptic_path, dataset_name, sequence_idx, output_path):
    seq_idxs = range(int(sequence_idx))

    img_dirs = glob(osp.join(panoptic_path, dataset_name, 'hdImgs', '*'))
    info_list = list()
    for img_dir in img_dirs:
        cam_name = osp.basename(img_dir)

        calib = None
        calib_file = f'{panoptic_path}/{dataset_name}/calibration_{dataset_name}.json'
        with open(calib_file) as f:
            calib = json.load(f)

        cam = None  # type: dict
        # for cam in calib['cameras']:
        #     if cam['name'] == cam_name:
        #         break

        cameras = {(cam['panel'],cam['node']):cam for cam in calib['cameras']}

        cam = cameras[(0,0)]

        K, R, t = np.array(cam['K']), np.array(cam['R']), np.array(cam['t'])
        resized_dir = osp.join(panoptic_path, 'processed', dataset_name,
                               '{0:02d}_{1:02d}'.format(cam['panel'], cam['node']))
        pathlib.Path(resized_dir).mkdir(parents=True, exist_ok=True)

        hd_img_path = osp.join(panoptic_path, dataset_name, 'hdImgs/')
        hd_skel_json_path = osp.join(panoptic_path, dataset_name, 'hdPose3d_stage1_coco19/')
        hd_face_json_path = osp.join(panoptic_path, dataset_name, 'hdFace3d/')
        hd_hand_json_path = osp.join(panoptic_path, dataset_name, 'hdHand3d/')
        
    
        for hd_idx in tqdm(seq_idxs):
            image_path = hd_img_path + '{0:02d}_{1:02d}/{0:02d}_{1:02d}_{2:08d}.jpg'.format(cam['panel'], cam['node'],
                                                                                            hd_idx)
            skel_json_fname = hd_skel_json_path + 'body3DScene_{0:08d}.json'.format(hd_idx)
            face_json_fname = hd_face_json_path+'faceRecon3D_hd{0:08d}.json'.format(hd_idx)
            hand_json_fname = hd_hand_json_path+'handRecon3D_hd{0:08d}.json'.format(hd_idx)

            with open(skel_json_fname) as dfile:
                bframe = json.load(dfile)
            with open(face_json_fname) as dfile:
                fframe = json.load(dfile)
            with open(hand_json_fname) as dfile:
                hframe = json.load(dfile)

            bboxes = list()
            gt_kpts2d = list()
            gt_face2d = list()
            gt_left2d = list()
            gt_right2d = list()

            for body in bframe['bodies']:
                skel = np.array(body['joints19']).reshape((-1, 4)).transpose()
                kpts3d_J24 = np.zeros((25, 4))
                kpts3d_J24[:19,:] = skel.T[kpts_coco19]
                valid = skel[3, :] > 0.1
                pt_J24 = projectPoints(kpts3d_J24.T[0:3, :], np.array(cam['K']), cam['R'], cam['t'], cam['distCoef'])
                kpts2d_J24 = np.zeros((25, 3))
                kpts2d_J24[:, :2] = pt_J24[:2].T
                kpts2d_J24[:, -1] = kpts3d_J24[:, -1]
                s_kpts2d = np.zeros_like(kpts2d_J24)
                s_kpts2d[..., -1] = kpts2d_J24[..., -1]
                s_kpts2d[..., :-1] = kpts2d_J24[..., :-1]
                
                valid_kps = s_kpts2d[s_kpts2d[:, 2] > 0, :2]
                if len(valid_kps) == 0:
                    tqdm.write(f"An invalid 2D pose found in {osp.basename(image_path)}")
                    continue
                s_kpts2d = s_kpts2d.flatten()
                gt_kpts2d.append(s_kpts2d.tolist())

            for face in fframe['people']:
                face3d = np.array(face['face70']['landmarks']).reshape((-1,3)).transpose()
                face3d_4 = np.concatenate((face3d[0,:].reshape((1,70)),face3d[1,:].reshape((1,70)),face3d[2,:].reshape((1,70)),np.ones(shape=(1,70))),axis=0)
                kpts3d_J24 = np.zeros((70, 4))
                kpts3d_J24 = face3d_4.T
                kpts3d_J24[:, -1] = kpts3d_J24[:, -1] > 0.1
                valid = skel[3, :] > 0.1
                pt_J24 = projectPoints(kpts3d_J24.T[0:3, :], np.array(cam['K']), cam['R'], cam['t'], cam['distCoef'])
                kpts2d_J24 = np.zeros((70, 3))
                kpts2d_J24[:, :2] = pt_J24[:2].T
                kpts2d_J24[:, -1] = kpts3d_J24[:, -1]
                s_kpts2d = np.zeros_like(kpts2d_J24)
                s_kpts2d[..., -1] = kpts2d_J24[..., -1]
                s_kpts2d[..., :-1] = kpts2d_J24[..., :-1]
                
                valid_kps = s_kpts2d[s_kpts2d[:, 2] > 0, :2]
                if len(valid_kps) == 0:
                    tqdm.write(f"An invalid 2D pose found in {osp.basename(image_path)}")
                    continue
                s_kpts2d = s_kpts2d.flatten()
                gt_face2d.append(s_kpts2d.tolist())

            for hand in hframe['people']:
                r_hand3d = np.array(hand['right_hand']['landmarks']).reshape((-1,3)).transpose()
                r_hand3d_4 = np.concatenate((r_hand3d[0,:].reshape((1,21)),r_hand3d[1,:].reshape((1,21)),r_hand3d[2,:].reshape((1,21)),np.ones(shape=(1,21))),axis=0)
                kpts3d_J24 = np.zeros((21, 4))
                kpts3d_J24 = r_hand3d_4.T
                kpts3d_J24[:, -1] = kpts3d_J24[:, -1] > 0.1
                valid = skel[3, :] > 0.1
                pt_J24 = projectPoints(kpts3d_J24.T[0:3, :], np.array(cam['K']), cam['R'], cam['t'], cam['distCoef'])
                kpts2d_J24 = np.zeros((21, 3))
                kpts2d_J24[:, :2] = pt_J24[:2].T
                kpts2d_J24[:, -1] = kpts3d_J24[:, -1]
                s_kpts2d = np.zeros_like(kpts2d_J24)
                s_kpts2d[..., -1] = kpts2d_J24[..., -1]
                s_kpts2d[..., :-1] = kpts2d_J24[..., :-1]
                
                valid_kps = s_kpts2d[s_kpts2d[:, 2] > 0, :2]
                if len(valid_kps) == 0:
                    tqdm.write(f"An invalid 2D pose found in {osp.basename(image_path)}")
                    continue
                s_kpts2d = s_kpts2d.flatten()
                gt_right2d.append(s_kpts2d.tolist())

                lhand3d = np.array(hand['left_hand']['landmarks']).reshape((-1,3)).transpose()
                lhand3d_4 = np.concatenate((lhand3d[0,:].reshape((1,21)),lhand3d[1,:].reshape((1,21)),lhand3d[2,:].reshape((1,21)),np.ones(shape=(1,21))),axis=0)
                kpts3d_J24 = np.zeros((21, 4))
                kpts3d_J24 = lhand3d_4.T
                kpts3d_J24[:, -1] = kpts3d_J24[:, -1] > 0.1
                valid = skel[3, :] > 0.1
                pt_J24 = projectPoints(kpts3d_J24.T[0:3, :], np.array(cam['K']), cam['R'], cam['t'], cam['distCoef'])
                kpts2d_J24 = np.zeros((21, 3))
                kpts2d_J24[:, :2] = pt_J24[:2].T
                kpts2d_J24[:, -1] = kpts3d_J24[:, -1]
                s_kpts2d = np.zeros_like(kpts2d_J24)
                s_kpts2d[..., -1] = kpts2d_J24[..., -1]
                s_kpts2d[..., :-1] = kpts2d_J24[..., :-1]
                
                valid_kps = s_kpts2d[s_kpts2d[:, 2] > 0, :2]
                if len(valid_kps) == 0:
                    tqdm.write(f"An invalid 2D pose found in {osp.basename(image_path)}")
                    continue
                s_kpts2d = s_kpts2d.flatten()
                gt_left2d.append(s_kpts2d.tolist())

            cur_info = {"people": [{
                            "person_id":[str(-1)],
                            "pose_keypoints_2d": gt_kpts2d[0],
                            "face_keypoints_2d": gt_face2d[0],
                            "hand_left_keypoints_2d": gt_left2d[0],
                            "hand_right_keypoints_2d": gt_right2d[0],
                            "pose_keypoints_3d": [],
                            "face_keypoints_3d": [],
                            "hand_left_keypoints_3d" : [],
                            "hand_right_keypoints_3d":[]
                        }],
                        'version': str(1.3)
                        }
            annotation_path = output_path + '/{0:02d}_{1:02d}_{2:08d}_keypoints.json'.format(cam['panel'], cam['node'],hd_idx)

            with open(annotation_path, 'w') as f:
                json.dump(cur_info, f)


if __name__ == '__main__':
    args = parser.parse_args()
    process_panoptic(args.panoptic_path, args.dataset_name, args.sequence_idx, args.output_path)
