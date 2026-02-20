import argparse
from functools import partial
import os
import os.path as osp
import time
import cv2
import json
import numpy as np
import sys
from util.camera import Camera
from Tracker.PoseTracker import Detection_Sample, PoseTracker,TrackState
from tqdm import tqdm
import copy

def main():
    scene_name = sys.argv[1]
    no_reid_merge = '--no-reid-merge' in sys.argv
    
    current_file_path = os.path.abspath(__file__)
    path_arr = current_file_path.split('/')[:-2]
    root_path = '/'.join(path_arr)
    
    det_dir = osp.join(root_path,"result/detection", scene_name)
    reid_dir = osp.join(root_path,"result/reid", scene_name)
    
    cal_dir = osp.join(root_path,'dataset/test', scene_name)
    save_dir = os.path.join(root_path,"result/track")
    save_path = osp.join(save_dir, scene_name+".txt")
    # if os.path.exists(save_path):
    #     print("exit",scene_name)
    #     exit()
    
    cams = sorted(os.listdir(cal_dir))
    cams = [c for c in cams if c.startswith("camera_")]
    cals = []
    for cam in cams:
        cals.append(Camera(osp.join(cal_dir,cam,"calibration.json")))

    det_data=[]
    reid_data=[]
    for cam in cams:
        # Detection
        det_path = osp.join(det_dir, cam + ".txt")
        if osp.exists(det_path):
            d = np.loadtxt(det_path, delimiter=",")
            if d.ndim < 2:
                d = np.zeros((0, 7))
        else:
            d = np.zeros((0, 7))
        det_data.append(d)

        # ReID
        reid_path = osp.join(reid_dir, cam + ".npy")
        if osp.exists(reid_path):
            reid_data_scene = np.load(reid_path, mmap_mode='r')
            if len(reid_data_scene):
                reid_data_scene = reid_data_scene / np.linalg.norm(reid_data_scene, axis=1, keepdims=True)
        else:
            reid_data_scene = np.zeros((0, 1024))
        reid_data.append(reid_data_scene)

    print("reading finish")

    
    max_frame = []
    for det_sv in det_data:
        if len(det_sv):
            max_frame.append(np.max(det_sv[:,0]))
    max_frame = int(np.max(max_frame))

    tracker = PoseTracker(cals, no_reid_merge=no_reid_merge)
    if no_reid_merge:
        print("*** ReID merge DISABLED — producing clean tracklets ***")
        save_path = osp.join(save_dir, scene_name + "_tracklets.txt")
    box_thred = 0.3
    results = []

    for frame_id in tqdm(range(max_frame+1),desc = scene_name):
        detection_sample_mv = []
        for v in range(tracker.num_cam):
            detection_sample_sv = []
            det_sv = det_data[v]
            if len(det_sv)==0:
                detection_sample_mv.append(detection_sample_sv)
                continue
            idx = det_sv[:,0]==frame_id
            cur_det = det_sv[idx]
            cur_reid = reid_data[v][idx]

            for det, reid in zip(cur_det, cur_reid):
                if det[-1]<box_thred or len(det)==0:
                    continue
                new_sample = Detection_Sample(bbox=det[2:], reid_feat=reid, cam_id = v, frame_id=frame_id)
                detection_sample_sv.append(new_sample)
            detection_sample_mv.append(detection_sample_sv)


        print("frame {}".format(frame_id),"det nums: ",[len(L) for L in detection_sample_mv])

        tracker.mv_update_wo_pred(detection_sample_mv, frame_id)
        #print(len([t for t in tracker.tracks if t.state == TrackState.Confirmed]))

        frame_results = tracker.output(frame_id)
        results += frame_results
        
        
    results = np.concatenate(results,axis=0)
    sort_idx = np.lexsort((results[:,2],results[:,0]))
    results = np.ascontiguousarray(results[sort_idx])
    np.savetxt(save_path, results)


if __name__ == '__main__':
    main()

        






    

    






    


