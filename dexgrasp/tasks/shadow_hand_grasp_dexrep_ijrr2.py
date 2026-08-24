# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import os
import os.path as osp
import numpy as np
import torch
import trimesh
from glob import glob
from tqdm import tqdm
from utils.torch_jit_utils import *
from tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi
from dexrep.ShareDexRepSensor import SharedDexRepSensor as DexRepEncoder
from dexrep.ShareDexRepSensor import SharedPnGSensor as PnGEncoder
from scipy.spatial.transform import Rotation as R
import open3d as o3d
# from dexgrasp.utils.update_params import update_seq_grasp_direction_batch
from utils.update_params import update_seq_grasp_direction_batch

# from pytorch3d.transforms import quaternion_to_matrix,euler_angles_to_matrix
from utils.hand_model import HandModel

# from ActionDiffusion.utils.vis_utils import html_antmation_save
from dexgrasp.utils.traj_utils import modify_hand_trajectory,downsampling_trajectory,compute_h2o_minimum_vec,\
    rotate_trajs_and_object_to_zneg_vectorized, unwrap_euler_batch_vectorized

from tasks.hand_base.shadow_grasp_base_task import ShadowHandGraspDexRepBase

_DexRepEncoder_Map = {
            'pnG': PnGEncoder,
            'DexRep': DexRepEncoder,
            'DexRep_debug': DexRepEncoder,
        }

class ShadowHandGraspDexRepIjrr2(ShadowHandGraspDexRepBase):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless,is_multi_agent=False, npy_list=None):
        self.cfg = cfg
        self.device_type = device_type
        self.device_id = device_id
        self.tips_idxs = [8, 12, 16, 21]
        self.big_tips_idx = [27]
        self.batch_load_data_dict(npy_list)
        cfg['env']['numEnvs'] = self.obj_trajs_info['grasp_seqs'].shape[0]

        super(ShadowHandGraspDexRepIjrr2,self).__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

        model_base_path = "../assets/mjcf/"
        self.hand_model = HandModel(
            mjcf_path=model_base_path + 'shadow_hand_vis_new.xml', mesh_path=model_base_path + 'meshes',
            contact_points_path=model_base_path + 'contact_points.json',
            penetration_points_path=model_base_path + 'penetration_points.json',
            n_surface_points=512,
            device='cpu',  # device_type #"cpu"
            use_joint21=True
        )
        a = 1


    def batch_load_data_dict(self, npy_list):
        self.object_code_list = []
        self.object_idxs = []
        self.obj_trajs_info = {'obj_scale':[], 'obj_rotmat':[],'grasp_seqs':[]}

        has_obj_code_idx = 'obj_code_idx' in npy_list[0]
        if has_obj_code_idx:
            self.obj_trajs_info['obj_code_idx'] = []

        for obj_id, data_dct in enumerate(npy_list):
            obj_code = data_dct['obj_code']
            self.object_code_list.append(obj_code)

            obj_trajs_info, object_idxs = self.load_data_dict(data_dct, [obj_id])
            self.object_idxs+=object_idxs
            for key, val in obj_trajs_info.items():
                self.obj_trajs_info[key].append(val)

        for key, val in self.obj_trajs_info.items():
            if key == 'obj_code_idx':
                self.obj_trajs_info[key] = np.array(val)
            else:
                self.obj_trajs_info[key] = np.concatenate(val, axis=0)

        # self.data_process()
        self.grasp_seqs = torch.tensor(self.obj_trajs_info['grasp_seqs'], device=f"{self.device_type}:{self.device_id}")

    def data_process(self):
        self.obj_trajs_info['grasp_seqs'] = unwrap_euler_batch_vectorized(self.obj_trajs_info['grasp_seqs'])
        self.obj_trajs_info['grasp_seqs'][:, 41:, :] = self.apply_fingers_grip(self.obj_trajs_info['grasp_seqs'][:, 41:, :],delta_angle=0.3)

        if self.cfg['env']['traj_modify']:
            self.obj_trajs_info['grasp_seqs'][:,:40,:] = modify_hand_trajectory(self.obj_trajs_info['grasp_seqs'][:,:40,:])
            a=1

        if self.cfg['env']['traj_down_sample']:
            self.obj_trajs_info['grasp_seqs'] = downsampling_trajectory(self.obj_trajs_info['grasp_seqs'])


        if self.cfg['env']['seq_start_pos_uniform'] and self.cfg['env']['env_mode']=='extract_obs':
            grasp_seqs = torch.from_numpy(self.obj_trajs_info['grasp_seqs'])

            if self.cfg['env']['seq_start_rot_uniform']:
                grasp_seqs, R_align = rotate_trajs_and_object_to_zneg_vectorized(grasp_seqs)
                obj_rotmat = self.obj_trajs_info['obj_rotmat']  # (N, 3, 3)
                self.obj_trajs_info['obj_rotmat'] = np.matmul(R_align.numpy(), obj_rotmat)  # (N, 3, 3)
                a=1

            grasp_seqs, Rz = update_seq_grasp_direction_batch(grasp_seqs) #(N,T, 28) (N, 3, 3)
            self.obj_trajs_info['grasp_seqs'] = grasp_seqs.numpy()
            # grasp_seqs = grasp_seqs.to(torch.device(self.cfg["device_type"] + f":{device_id}"))

            obj_rotmat = self.obj_trajs_info['obj_rotmat']  # (N, 3, 3)
            self.obj_trajs_info['obj_rotmat'] = np.matmul(Rz.numpy(), obj_rotmat)  # (N, 3, 3)
            a = 1
        else:
            grasp_seqs = torch.from_numpy(self.obj_trajs_info['grasp_seqs'])#.to(torch.device(self.cfg["device_type"] + f":{device_id}"))

        self.grasp_seqs =  grasp_seqs.to(torch.device(self.device_type + f":{self.device_id}"))


    def get_obj_idx_mask(self, idx):
        return (torch.tensor(self.object_idxs) == idx)

    def load_data_dict(self, data_dict, obj_id=None):
        data_dict.pop('obj_code')
        obj_trajs_info = data_dict

        if isinstance(obj_id,list):
            object_idxs = obj_id*obj_trajs_info['grasp_seqs'].shape[0]
        else:
            object_idxs = [0]* obj_trajs_info['grasp_seqs'].shape[0]

        return obj_trajs_info, object_idxs


