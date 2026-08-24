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
import random
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

from pytorch3d.transforms import quaternion_to_matrix,euler_angles_to_matrix
from utils.hand_model import HandModel
from utils.HandModel_linkerhand import HandModel_Linkerhand
from ActionDiffusion.utils.vis_utils import html_antmation_save
from dexgrasp.utils.traj_utils import modify_hand_trajectory,downsampling_trajectory,compute_h2o_minimum_vec,\
    rotate_trajs_and_object_to_zneg_vectorized, unwrap_euler_batch_vectorized

_DexRepEncoder_Map = {
            'pnG': PnGEncoder,
            'DexRep': DexRepEncoder,
            'DexRep_debug': DexRepEncoder,
        }

def list_object_codes_from_dir(trajs_path):
    npy_paths = sorted(glob(osp.join(trajs_path, "*.npy")))
    if not npy_paths:
        raise FileNotFoundError("no npy files found in trajs_path: {}".format(trajs_path))
    return [osp.splitext(osp.basename(path))[0] for path in npy_paths]


def should_scan_object_codes(object_code_dict):
    return (
        object_code_dict is None
        or object_code_dict == "auto"
        or object_code_dict == ["auto"]
        or len(object_code_dict) == 0
    )

class o6HandGraspDexRepIjrr(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless,
                 agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False, npy_list=None):

        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index
        self.is_multi_agent = is_multi_agent
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]
        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations
        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]
        self.o6_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.actions_are_normalized = self.cfg["env"].get("actions_are_normalized", False)
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)

        # self.tips_idxs = [8,12,16,21]
        # self.big_tips_idx = [27]
        self.tips_idxs = [8, 9, 10, 11]
        self.big_tips_idx = [7]
        self.per_obj_seq_idx=None
        self.object_idxs = [0]
        print("Averaging factor: ", self.av_factor)

        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]

        # model_base_path = "../assets/mjcf/"
        # self.hand_model = HandModel(
        #     mjcf_path=model_base_path + 'shadow_hand_vis_new.xml', mesh_path=model_base_path + 'meshes',
        #     contact_points_path=model_base_path + 'contact_points.json',
        #     penetration_points_path=model_base_path + 'penetration_points.json',
        #     n_surface_points=512,
        #     device='cpu',#device_type #"cpu"
        #     use_joint21=True
        # )
        # model_base_path = "../assets/linkerhand/o6/right"
        self.hand_model = HandModel_Linkerhand(
        robot_name="linkerhand",
        urdf_filename="linkerhand_o6_right.urdf",
        mesh_path="",
        batch_size=1,
        device='cpu',
        mesh_nsp=128,
        hand_scale=1.0,
        asset_dir="../assets/linkerhand/o6/right",
        allow_missing_contacts=True,
    )
        a=1



        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)
        self.obs_type = self.cfg["env"]["observationType"]
        self.o6_policy_obs_mode = self.cfg["env"].get("o6_policy_obs_mode", "prev_action_obj_rot")
        print("Obs type:", self.obs_type)

        num_obs = 236 + 64
        dexrep_obs_dim = (
            int(self.cfg["env"]["obs_dim"]["prop"])
            + int(self.cfg["env"]["obs_dim"]["dexrep_sensor"])
            + int(self.cfg["env"]["obs_dim"]["dexrep_pnl"])
        )
        self.num_obs_dict = {
            "full_state": num_obs,
            "DexRep": dexrep_obs_dim if self.o6_policy_obs_mode == "dexrep" else 21,
            "obs_pcds": (1024, 3)
        }
        # if use DexRep Encoder
        if self.obs_type in _DexRepEncoder_Map.keys():
            assert "dexrep" in cfg.keys()
            self.use_dexrep = True
            self.use_pnG = False

            self.DexRepEncoder = _DexRepEncoder_Map[self.obs_type](cfg, device_type + f":{device_id}")

        elif self.obs_type == 'obs_pcds' or self.obs_type == 'obs_h2o':
            self.use_dexrep = False
            self.use_pnG = False
            self.use_geodex = False
            self.use_obs_pcds = True
        else:
            self.use_dexrep = False

        if self.cfg['env']['bc_model_name'] == 'ActorCriticPNG':
            self.use_dexrep = False
            self.use_pnG = True

        self.num_hand_obs = 66 + 95 + 24 + 6  # 191 =  22*3 + (65+30) + 24
        self.up_axis = 'z'
        # self.fingertips = ["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal",
        #                    "robot0:thdistal"]
        self.fingertips = ["rh_thumb_distal", "rh_index_distal", "rh_middle_distal",
                           "rh_ring_distal", "rh_pinky_distal"]
        self.hand_center = ["robot0:palm"]
        self.num_fingertips = len(self.fingertips)
        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]
        num_states = 0
        if self.asymmetric_obs:
            num_states = 211
        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        # self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numStates"] = 21
        self.num_agents = 1
        # self.cfg["env"]["numActions"] = 28
        self.cfg["env"]["numActions"] = 12
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        # self.dexrep_hand = [
        #     "robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal", "robot0:thdistal",
        #     "robot0:ffmiddle", "robot0:mfmiddle", "robot0:rfmiddle", "robot0:lfmiddle", "robot0:thmiddle",
        #     "robot0:ffproximal", "robot0:mfproximal", "robot0:rfproximal", "robot0:lfmetacarpal", "robot0:thproximal"
        # ]
        self.dexrep_hand = [
            "rh_thumb_distal", "rh_index_distal", "rh_middle_distal", "rh_ring_distal", "rh_pinky_distal",]

        if self.cfg['env']['obj_type'] in ['seen', 'one']:
            trajs_path = self.cfg['trajs_path']['train']
        else:
            trajs_path = self.cfg['trajs_path']['valid']
        if os.environ.get("DEXGRASP_INFER_DATA_DIR") and cfg['env']['env_mode'] == 'bc_env_infer':
            trajs_path = os.environ["DEXGRASP_INFER_DATA_DIR"]

        if cfg['env']['env_mode']=='rl_learning':
            self.obj_trajs_info = self.rl_data_load(trajs_path)

        else:
            if npy_list is None:
                if should_scan_object_codes(self.cfg['env'].get('object_code_dict', [])):
                    self.cfg['env']['object_code_dict'] = list_object_codes_from_dir(trajs_path)
                npy_list = []
                for obj_id in self.cfg['env']['object_code_dict']:
                    npy_path = os.path.join(trajs_path, f"{obj_id}.npy")
                    data = np.load(npy_path, allow_pickle=True).item()
                    if 'obj_code' not in data:
                        data = dict(data)
                        data['obj_code'] = obj_id
                    npy_list.append(data)
            else:
                object_codes = [data['obj_code'] for data in npy_list if 'obj_code' in data]
                if len(object_codes) == len(npy_list):
                    self.cfg['env']['object_code_dict'] = object_codes
            self.batch_load_data_dict(npy_list)

        self.obj_trajs_info['grasp_seqs'] = unwrap_euler_batch_vectorized( self.obj_trajs_info['grasp_seqs'])

        self.cfg['env']['numEnvs'] = self.obj_trajs_info['grasp_seqs'].shape[0]
        self.table_height = self.cfg['env']['table_height']

        if self.cfg['env']['traj_modify']:
            self.obj_trajs_info['grasp_seqs'][:,:40,:] = modify_hand_trajectory(self.obj_trajs_info['grasp_seqs'][:,:40,:])
            a=1

        if  self.cfg['env']['traj_down_sample']:
            self.obj_trajs_info['grasp_seqs'] = downsampling_trajectory( self.obj_trajs_info['grasp_seqs'])

        self.Rz = None

        if cfg['env']['seq_start_pos_uniform'] and cfg['env']['env_mode']=='extract_obs':
            grasp_seqs = torch.from_numpy(self.obj_trajs_info['grasp_seqs'])

            if cfg['env']['seq_start_rot_uniform']:
                grasp_seqs, R_align = rotate_trajs_and_object_to_zneg_vectorized(grasp_seqs)
                obj_rotmat = self.obj_trajs_info['obj_rotmat']  # (N, 3, 3)
                self.obj_trajs_info['obj_rotmat'] = np.matmul(R_align.numpy(), obj_rotmat)  # (N, 3, 3)

            grasp_seqs, Rz = update_seq_grasp_direction_batch(grasp_seqs) #(N,T, 28) (N, 3, 3)

            self.obj_trajs_info['grasp_seqs'] = grasp_seqs.numpy()
            self.grasp_seqs = grasp_seqs.to(torch.device(device_type + f":{device_id}"))

            obj_rotmat = self.obj_trajs_info['obj_rotmat'] #(N, 3, 3)
            self.obj_trajs_info['obj_rotmat'] = np.matmul(Rz.numpy(), obj_rotmat) #(N, 3, 3)
        else:
            self.grasp_seqs = torch.from_numpy(self.obj_trajs_info['grasp_seqs']).to(
                torch.device(device_type + f":{device_id}"))

        infer_runtime = self.cfg['env'].get('infer_runtime', {}) or {}
        enable_camera_sensors = bool(
            infer_runtime.get(
                'enable_camera_sensors',
                self.cfg['env'].get('enable_camera_sensors', False),
            )
        )
        if infer_runtime.get("record_video", False) or os.environ.get("DEXGRASP_RECORD_VIDEO") == "1":
            enable_camera_sensors = True
        if os.environ.get("DEXGRASP_ENABLE_CAMERA_SENSORS") is not None:
            enable_camera_sensors = os.environ["DEXGRASP_ENABLE_CAMERA_SENSORS"].lower() in {
                "1", "true", "yes", "on"
            }

        super().__init__(cfg=self.cfg, enable_camera_sensors=enable_camera_sensors)

        self.num_dexrep_hand = len(self.dexrep_hand)
        if self.viewer != None:
            cam_pos = gymapi.Vec3(-1.0, -1.0, 1.5)
            cam_target = gymapi.Vec3(0.0, 0.0, 1.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # if self.obs_type == "full_state" or self.asymmetric_obs:
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs,
                                                self.num_o6_hand_dofs + self.num_object_dofs)
        self.dof_force_tensor = self.dof_force_tensor[:, :self.num_o6_hand_dofs]


        self.sim_refresh()


        self.z_theta = torch.zeros(self.num_envs, device=self.device)

        # create some wrapper tensors for different slices
        self.o6_hand_default_dof_pos = torch.zeros(self.num_o6_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.o6_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_o6_hand_dofs]
        self.o6_hand_dof_pos = self.o6_hand_dof_state[..., 0]
        self.o6_hand_dof_vel = self.o6_hand_dof_state[..., 1]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone()
        self.saved_root_tensor[self.object_indices, 9:10] = 0.0
        infer_runtime = self.cfg['env'].get('infer_runtime', {}) or {}
        if infer_runtime.get("reset_root_from_saved_tensor", False):
            self.object_init_state = self.saved_root_tensor[self.object_indices].clone()
            self.object_init_state[:, 7:13] = 0.0
            self.goal_init_state = self.saved_root_tensor[self.goal_object_indices].clone()
            self.goal_init_state[:, 7:13] = 0.0
            self.goal_states = self.goal_init_state.clone()
            self.goal_pose = self.goal_states[:, 0:7]
            self.goal_pos = self.goal_states[:, 0:3]
            self.goal_rot = self.goal_states[:, 3:7]
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs,-1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.current_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.total_successes = 0
        self.total_resets = 0

        self.pre_target_actions = torch.zeros((self.num_envs, 12), device=self.device, dtype=torch.float)
        self.check_mask = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)
        self.iter_check_store = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)

    def batch_load_data_dict(self, npy_list):
        # self.object_code_list = []
        self.object_idxs = []
        self.obj_trajs_info = {'obj_scale':[], 'obj_rotmat':[],'grasp_seqs':[]}
        test_num = int(os.environ.get("DEXGRASP_TEST_NUM", self.cfg['env'].get('test_num', 0)))
        max_envs = int(os.environ.get("DEXGRASP_NUM_ENVS", self.cfg['env'].get('numEnvs', 0)))
        per_object_limit = min([value for value in [test_num, max_envs] if value > 0], default=0)

        for obj_id, data_dct in enumerate(npy_list):
            # obj_code = data_dct['obj_code']
            # self.object_code_list.append(obj_code)

            obj_trajs_info, object_idxs = self.load_data_dict(data_dct, [obj_id])
            if per_object_limit > 0 and obj_trajs_info['grasp_seqs'].shape[0] > per_object_limit:
                obj_trajs_info = dict(obj_trajs_info)
                for key in self.obj_trajs_info:
                    obj_trajs_info[key] = obj_trajs_info[key][:per_object_limit]
                object_idxs = object_idxs[:per_object_limit]
            self.object_idxs+=object_idxs
            # for key, val in obj_trajs_info.items():
            #     self.obj_trajs_info[key].append(val)
            for key in self.obj_trajs_info:
                # if key in obj_trajs_info:
                self.obj_trajs_info[key].append(obj_trajs_info[key])

        for key, val in self.obj_trajs_info.items():
            self.obj_trajs_info[key] = np.concatenate(val,axis=0)
        if self.cfg['env'].get('prepend_dummy_env0', False) and self.obj_trajs_info['grasp_seqs'].shape[0] > 0:
            for key in self.obj_trajs_info:
                self.obj_trajs_info[key] = np.concatenate(
                    [self.obj_trajs_info[key][0:1].copy(), self.obj_trajs_info[key]],
                    axis=0,
                )
            self.object_idxs = [self.object_idxs[0]] + self.object_idxs
            print("prepend_dummy_env0: duplicated first trajectory as env0 and excluded it from metrics")
        a=1

    def load_data_dict(self, data_dict, obj_id=None):
        # data_dict.pop('obj_code')
        obj_trajs_info = data_dict

        if isinstance(obj_id,list):
            object_idxs = obj_id*obj_trajs_info['grasp_seqs'].shape[0]
        else:
            object_idxs = [0]* obj_trajs_info['grasp_seqs'].shape[0]

        return obj_trajs_info, object_idxs

    def rl_data_load(self, trajs_path):
        data_npt_list = glob(osp.join(trajs_path, "**.npy"))
        data_npt_list = sorted( [f for f in data_npt_list if os.path.getsize(f) >= 1024],key=lambda path: os.path.getsize(path),reverse=True)
        data_npt_list = data_npt_list[17:18]
        traj_data_dct = {'grasp_seqs':[], 'obj_rotmat':[], 'obj_scale':[]}

        self.cfg['env']['object_code_dict'] = []
        self.per_obj_seq_idx = {}
        self.object_idxs = []
        seq_idx=0
        for obj_id, path in enumerate(tqdm(data_npt_list, desc="Processing")):
            obj_code_name = osp.basename(path).split('.')[0]

            np_data_dct = np.load(path, allow_pickle=True).item()
            # obj_codes = [obj_code_name]*len(np_data_dct['grasp_seqs'])
            # self.cfg['env']['object_code_dict']+=obj_codes

            self.cfg['env']['object_code_dict'].append(obj_code_name)

            seq_num = len(np_data_dct['grasp_seqs'])
            self.per_obj_seq_idx[obj_id]=[seq_idx,seq_idx+seq_num]
            self.object_idxs+=[obj_id]*len(np_data_dct['grasp_seqs'])
            seq_idx +=seq_num
            a=1
            for key in traj_data_dct.keys():
                val = np_data_dct[key]
                traj_data_dct[key].append(val)


        for  key, val in traj_data_dct.items():
            traj_data_dct[key] = np.concatenate(val)


        print('RL Data Load Finish ObjNum={} -- SeqNum={}'.format(len(self.cfg['env']['object_code_dict']), seq_idx))
        return traj_data_dct


    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)


    def _create_envs(self, num_envs, spacing, num_per_row):

        object_code_list = self.cfg['env']['object_code_dict']
        self.object_code_list = object_code_list
        # all_scales = set()

        self.repose_z = self.cfg['env']['repose_z']

        self.grasp_data = {}
        assets_path = '../assets'
        print(f'Num Objs: {len(self.object_code_list)}')
        print(f'Num Envs: {self.num_envs}')

        self.goal_cond = self.cfg["env"]["goal_cond"]
        self.random_prior = self.cfg['env']['random_prior']
        self.random_time = self.cfg["env"]["random_time"]
        self.target_qpos = torch.zeros((self.num_envs, 22), device=self.device)
        self.target_hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_hand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_init_euler_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.object_init_z = torch.zeros((self.num_envs, 1), device=self.device)

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        o6_hand_asset, o6_hand_dof_props, table_texture_handle = self._load_o6_hand_asset()

        goal_asset_dict, object_asset_dict = self._load_object_asset(assets_path)

        # create table asset
        table_asset, table_dims = self._load_table_asset()

        o6_hand_start_pose = gymapi.Transform()
        hand_root_pos = self.cfg["env"].get("hand_root_pos", [0.0, 0.0, 0.0])
        o6_hand_start_pose.p = gymapi.Vec3(*hand_root_pos)
        self.init_hand_pos_z = float(hand_root_pos[2])
        o6_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(0, -1.57, 0)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.0, 0.0, 0)  # gymapi.Vec3(0.0, 0.0, 0.72)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)
        pose_dx, pose_dy, pose_dz = -1.0, 0.0, -0.0

        self.goal_displacement = gymapi.Vec3(-0., 0.0, 0.3+self.table_height)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement
        goal_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)

        goal_start_pose.p.z -= 0.0

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)

        # compute aggregate size
        # max_agg_bodies = self.num_o6_hand_bodies * 1 + 2 * self.num_object_bodies + 1  ##
        # max_agg_shapes = self.num_o6_hand_shapes * 1 + 2 * self.num_object_shapes + 1  ##

        self.o6_hands = []
        self.objects = []
        self.envs = []
        self.object_init_state = []
        self.goal_init_state = []
        self.hand_start_states = []
        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.table_indices = []
        self.dexrep_hand_indices = []
        for o in range(len(self.dexrep_hand)):
            dexrep_hand_env_handle = self.gym.find_asset_rigid_body_index(o6_hand_asset, self.dexrep_hand[o])
            self.dexrep_hand_indices.append(dexrep_hand_env_handle)
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(o6_hand_asset, name) for name in self.fingertips]

        body_names = {
            'wrist': 'rh_hand_base_link',
            'palm': 'rh_hand_base_link',
            'thumb': 'rh_thumb_distal',
            'index': 'rh_index_distal',
            'middle': 'rh_middle_distal',
            'ring': 'rh_ring_distal',
            'little': 'rh_pinky_distal'
        }
        self.hand_body_idx_dict = {}
        for name, body_name in body_names.items():
            self.hand_body_idx_dict[name] = self.gym.find_asset_rigid_body_index(o6_hand_asset, body_name)

        # create fingertip force sensors, if needed
        # if self.obs_type == "full_state" or self.asymmetric_obs:
        sensor_pose = gymapi.Transform()
        for ft_handle in self.fingertip_handles:
            self.gym.create_asset_force_sensor(o6_hand_asset, ft_handle, sensor_pose)

        # self.object_scale_buf = {}
        self.obj_half_height_list = []

        object_asset_cfg = self.cfg["env"]["asset"]
        self.asset_root = object_asset_cfg["assetRoot"]
        self.obj_asset_root = self.asset_root + object_asset_cfg["assetFileNameObj"]
        self.raw_obj_asset_root = self.asset_root + object_asset_cfg["assetFileNameObj_raw"]

        self.obj_mesh_list = []
        self.obj_sample_points_list = []
        for obj_code in self.object_code_list:
            dexrep_load = self.raw_obj_asset_root + obj_code + "/coacd" + f'/decomposed.obj'
            obj_mesh = trimesh.load_mesh(dexrep_load)
            if isinstance(obj_mesh, trimesh.Scene):
                obj_mesh = trimesh.util.concatenate([geometry for geometry in obj_mesh.geometry.values()])
            obj_sample_points = self.get_object_sample_points(obj_mesh)
            self.obj_mesh_list.append(obj_mesh)
            self.obj_sample_points_list.append(obj_sample_points)


        if len(self.object_code_list)==1:
            dexrep_load = self.raw_obj_asset_root + obj_code + "/coacd" + f'/decomposed.obj'
            obj_mesh = trimesh.load_mesh(dexrep_load)
            self.obj_mesh = obj_mesh
            self.obj_sample_points = self.get_object_sample_points(obj_mesh)

        down_sample = True if self.obs_type == 'obs_pcds' else False
        self.obj_init_obj_pcds = self.get_batch_obj_pcds(down_sample=down_sample).to(self.device)  # (B,N,3)
        self.pcds_max = torch.tensor([[0.2, 0.2, 0.5]], dtype=torch.float32).to(self.device)
        self.pcds_min = torch.tensor([[-0.2, -0.2, -0.3]], dtype=torch.float32).to(self.device)
        self.pcds_center = torch.tensor([[0.00341, 0.0231, 0.7782]], dtype=torch.float32).to(self.device)

        infer_runtime = self.cfg['env'].get('infer_runtime', {}) or {}
        if infer_runtime.get('skip_origin_env', False):
            dummy_env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            print("skip_origin_env: created one empty env before real envs")

        for i in range(self.num_envs):
            # object_idx_this_env = i % len(self.object_code_list)
            object_idx_this_env =self.object_idxs[i]
            object_code_this_env = self.object_code_list[object_idx_this_env]
            dexrep_load_this_env = self.raw_obj_asset_root + object_code_this_env + "/coacd" + f'/decomposed.obj'
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            max_agg_bodies = self.num_o6_hand_bodies + self.num_object_bodies_list[object_idx_this_env] + 2
            max_agg_shapes = self.num_o6_hand_shapes + self.num_object_shapes_list[object_idx_this_env] + 2

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # load o6 hand  for each env
            o6_hand_actor = self._load_o6_hand(env_ptr, i, o6_hand_asset, o6_hand_dof_props, o6_hand_start_pose)

            # load object for each env

            scale = self.obj_trajs_info['obj_scale'][i]
            obj_rotmat = self.obj_trajs_info['obj_rotmat'][i]

            obj_half_height, pcd = self.get_obj_half_height(scale, obj_rotmat, object_idx_this_env)
            # self.obj_half_height_list.append(obj_half_height)
            object_start_pose = self.set_obj_start_root_state(obj_half_height, obj_rotmat)
            object_handle = self._load_object(env_ptr, goal_start_pose, i, object_asset_dict, object_idx_this_env, object_start_pose, scale)
            # DexRep or pnG load object
            # assert len(self.object_code_list) == 1
            if self.use_dexrep:
                self.DexRepEncoder.load_cache_stl_file(
                    obj_idx=i,
                    obj_path=dexrep_load_this_env,
                    scale=scale)
            elif self.use_pnG:
                self.PnGEncoder.load_cache_stl_file(
                    obj_idx=i,
                    obj_path=dexrep_load_this_env,
                    scale=scale
                )
            elif self.use_geodex:
                self.GeoDexWrapper.load_cache_stl_file(
                    obj_idx=i,
                    obj_path=dexrep_load_this_env,
                    scale=scale
                )
            if self.use_dexrep:
                self.DexRepEncoder.load_batch_env_obj(object_idx_this_env)
            elif self.use_pnG:
                self.PnGEncoder.load_batch_env_obj(object_idx_this_env)
            elif self.use_geodex:
                self.GeoDexWrapper.load_batch_env_obj(object_idx_this_env)

            # add goal object
            # goal_asset_dict[id][scale_id]
            goal_handle = self.gym.create_actor(env_ptr, goal_asset_dict[object_idx_this_env], goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)
            self.gym.set_actor_scale(env_ptr, goal_handle, 1.0)

            # add table
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, -1, 0)
            self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)

            # ------------- set friction --------------
            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            table_shape_props[0].friction = 1
            if self.cfg['env']['random_obj_mass_friction']:
                friction = random.uniform(0.1, 1.0)
            else:
                friction = self.cfg['env']['obj_friction']
            object_shape_props[0].friction = friction #1
            # object_shape_props[0].friction = self.cfg['env']['obj_friction'] #1
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_shape_props)

            # ---------set mass -------------
            if self.cfg["env"]["set_obj_mass"]:
                object_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                if len(object_body_props)>=1:
                    if self.cfg["env"]['random_obj_mass_friction']:
                        obj_mass = random.uniform(0.05, 0.5)
                    else:
                        obj_mass = self.cfg["env"]["obj_mass"]
                    for object_body_prop in object_body_props:
                        # object_body_prop.mass = self.cfg["env"]["obj_mass"]/len(object_body_props)
                        object_body_prop.mass = obj_mass/len(object_body_props)
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, object_body_props)

            object_color = [90/255, 94/255, 173/255]
            self.gym.set_rigid_body_color(env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*object_color))
            table_color = [150/255, 150/255, 150/255]
            self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*table_color))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.o6_hands.append(o6_hand_actor)
            self.objects.append(object_handle)


        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object_init_state[:, 7:13] = 0.0
        self.goal_init_state = to_torch(self.goal_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_init_state[:, 7:13] = 0.0
        self.goal_states = self.goal_init_state.clone()
        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]
        self.goal_states[:, self.up_axis_idx] -= 0

        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)
        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)



    # def obj_pcd_transforms_from_state(self, obj_state, obj_verts):
    #     # B,D = obj_state.size()
    #
    #     obj_pos = obj_state[...,:3].reshape(-1,3)#(B,T,3)
    #     obj_quat = obj_state[...,3:][...,[3,0,1,2]].reshape(-1, 4)#(B*T,4)
    #     obj_rot = quaternion_to_matrix(obj_quat).float() #(B*T, 3, 3)
    #
    #
    #     obj_verts = obj_verts.unsqueeze(1).repeat(1,T,1,1).reshape(B*T, -1, 3) #(B*T,N,3)
    #     obj_verts = torch.bmm(obj_verts.float(), obj_rot.transpose(1, 2)) + obj_pos.unsqueeze(1)  # (B*T,N,3)
    #     obj_verts = obj_verts.reshape(B,T,-1, 3) # (B, T, N,3)
    #
    #     return obj_verts  # (B, T, N,3)

    def obj_pcd_transforms_from_state(self, obj_state, obj_verts):
        # B,D = obj_state.size()
        obj_pos = obj_state[...,:3].reshape(-1,3)#(B,3)
        obj_quat = obj_state[...,3:][...,[3,0,1,2]].reshape(-1,4)#(B,4)
        obj_rot = quaternion_to_matrix(obj_quat).float() #(B, 3, 3)
        obj_verts = torch.bmm(obj_verts.float(), obj_rot.transpose(1, 2)) + obj_pos.unsqueeze(1)  # (B,N,3)
        return obj_verts

    # def get_seq_object_mesh(self, obj_state, select_idxs=None):
    #     obj_scale = torch.tensor(self.obj_trajs_info['obj_scale']).unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
    #     obj_rotmat = torch.tensor(self.obj_trajs_info['obj_rotmat'])  # (B,3,3)
    #
    #     if select_idxs is not None:
    #         obj_state = obj_state[select_idxs]
    #         obj_scale = obj_scale[select_idxs] # (B,1,1)
    #         obj_rotmat = obj_rotmat[select_idxs]  # (B,3,3)
    #
    #     B,T,D = obj_state.size()
    #
    #     obj_verts = torch.tensor(self.obj_mesh.vertices,dtype=torch.float32).unsqueeze(0).repeat(B,1,1) #(B, N, 3)
    #     obj_verts = torch.bmm(obj_verts, obj_rotmat.transpose(1,2))*obj_scale  #(B, N, 3)
    #
    #
    #     obj_verts = self.obj_pcd_transforms_from_state(obj_state, obj_verts)  # (B, T, N,3)
    #
    #     # obj_pos = obj_state[...,:3].reshape(-1,3)#(B,T,3)
    #     # obj_quat = obj_state[...,3:][...,[3,0,1,2]].reshape(-1, 4)#(B*T,4)
    #     # obj_rot = quaternion_to_matrix(obj_quat).float() #(B*T, 3, 3)
    #     #
    #     # obj_verts = obj_verts.unsqueeze(1).repeat(1,T,1,1).reshape(B*T, -1, 3) #(B*T,N,3)
    #     # obj_verts = torch.bmm(obj_verts.float(), obj_rot.transpose(1, 2)) + obj_pos.unsqueeze(1)  # (B*T,N,3)
    #     # obj_verts = obj_verts.reshape(B,T,-1, 3) # (B, T, N,3)
    #
    #     obj_seq_mesh_list = []
    #     for i in range(B):
    #         list_i = [trimesh.Trimesh(vertices=obj_verts[i,j].numpy(), faces=self.obj_mesh.faces) for j in range(T)]
    #         obj_seq_mesh_list.append(list_i)
    #     return obj_seq_mesh_list

    def get_seq_object_mesh(self, obj_state, select_idxs=None):
        obj_scale = torch.tensor(self.obj_trajs_info['obj_scale']).unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
        obj_rotmat = torch.tensor(self.obj_trajs_info['obj_rotmat'])  # (B,3,3)

        if select_idxs is not None:
            obj_state = obj_state[select_idxs]
            obj_scale = obj_scale[select_idxs] # (B,1,1)
            obj_rotmat = obj_rotmat[select_idxs]  # (B,3,3)

        B,T,D = obj_state.size()

        obj_verts = torch.tensor(self.obj_mesh.vertices,dtype=torch.float32).unsqueeze(0).repeat(B,1,1) #(B, N, 3)
        obj_verts = torch.bmm(obj_verts, obj_rotmat.transpose(1,2))*obj_scale  #(B, N, 3)


        obj_verts = self.obj_pcd_seq_transforms_from_state(obj_state, obj_verts)  # (B, T, N,3)


        obj_seq_mesh_list = []
        for i in range(B):
            list_i = [trimesh.Trimesh(vertices=obj_verts[i,j].numpy(), faces=self.obj_mesh.faces) for j in range(T)]
            obj_seq_mesh_list.append(list_i)
        return obj_seq_mesh_list

    def obj_pcd_seq_transforms_from_state(self, obj_state, obj_verts):
        B,T,D = obj_state.size()

        obj_pos = obj_state[...,:3].reshape(-1,3)#(B,T,3)
        obj_quat = obj_state[...,3:][...,[3,0,1,2]].reshape(-1, 4)#(B*T,4)
        obj_rot = quaternion_to_matrix(obj_quat).float() #(B*T, 3, 3)


        obj_verts = obj_verts.unsqueeze(1).repeat(1,T,1,1).reshape(B*T, -1, 3) #(B*T,N,3)
        obj_verts = torch.bmm(obj_verts.float(), obj_rot.transpose(1, 2)) + obj_pos.unsqueeze(1)  # (B*T,N,3)
        obj_verts = obj_verts.reshape(B,T,-1, 3) # (B, T, N,3)

        return obj_verts  # (B, T, N,3)

    def get_seq_obj_pcd(self, obj_state=None, select_idxs=None):
        obj_original_sample_points= self.get_object_sample_points(self.obj_mesh, point_num=512) #(N,512)

        obj_scale = torch.tensor(self.obj_trajs_info['obj_scale']).unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
        obj_rotmat = torch.tensor(self.obj_trajs_info['obj_rotmat'])  # (B,3,3)

        if select_idxs is not None:
            obj_state = obj_state[select_idxs]
            obj_scale = obj_scale[select_idxs] # (B,1,1)
            obj_rotmat = obj_rotmat[select_idxs]  # (B,3,3)

        B,T,D = obj_state.size()

        obj_sample_points = torch.tensor(obj_original_sample_points,dtype=torch.float32).unsqueeze(0).repeat(B,1,1) #(B, N, 3)
        obj_sample_points = torch.bmm(obj_sample_points, obj_rotmat.transpose(1,2))*obj_scale  #(B, N, 3)

        obj_sample_points = self.obj_pcd_transforms_from_state(obj_state, obj_sample_points) #(B,T,  N, 3)
        return obj_sample_points

    # def get_batch_obj_pcds(self,down_sample=True):
    #     if down_sample:
    #         sample_points = self.get_down_sample_points(self.obj_sample_points) #（512，3）
    #     else:
    #         sample_points = self.obj_sample_points
    #     obj_scale = torch.tensor(self.obj_trajs_info['obj_scale']).unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
    #     obj_rotmat = torch.tensor(self.obj_trajs_info['obj_rotmat'])  # (B,3,3)
    #     B = obj_rotmat.size()[0]
    #
    #     obj_sample_points = torch.tensor(sample_points,dtype=torch.float32).unsqueeze(0).repeat(B,1,1) #(B, N, 3)
    #     obj_sample_points = torch.bmm(obj_sample_points, obj_rotmat.transpose(1,2))*obj_scale  #(B, N, 3)
    #     return obj_sample_points

    def get_batch_obj_pcds(self, down_sample=True):
        if down_sample:
            down_sampled_points_list = [self.get_down_sample_points(pcd) for pcd in
                                        self.obj_sample_points_list]  # list of (N, 3)
        else:
            down_sampled_points_list = self.obj_sample_points_list  # list of (N, 3)

        obj_scale = torch.tensor(self.obj_trajs_info['obj_scale']).unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
        obj_rotmat = torch.tensor(self.obj_trajs_info['obj_rotmat'])  # (B,3,3)
        object_idxs = torch.tensor(self.object_idxs)  # (B,)
        B = obj_rotmat.size(0)

        device = obj_rotmat.device

        obj_pcd_list = [
            torch.tensor(down_sampled_points_list[idx], dtype=torch.float32, device=device)
            for idx in object_idxs
        ]
        obj_sample_points = torch.stack(obj_pcd_list, dim=0)  # (B, N, 3)

        obj_sample_points = torch.bmm(obj_sample_points, obj_rotmat.transpose(1, 2)) * obj_scale  # (B, N, 3)

        return obj_sample_points

    def get_down_sample_points(self, pcd, n_num=512):
        indices = np.random.choice(pcd.shape[0], n_num, replace=False)
        return pcd[indices]

    def get_obs_pcds(self, obj_state, hand_state):
        obj_pcds =self.obj_pcd_transforms_from_state(obj_state,self.obj_init_obj_pcds) #(B,512,3)

        self.hand_model_forward(hand_state.cpu())
        hand_pcds,_ = self.hand_model.get_surface_points() #(B,512,3)

        obs_pcds = torch.cat([hand_pcds.to(obj_pcds.device),obj_pcds],dim=-2)#(B,1024,3)
        obs_pcds-=self.pcds_center
        obs_pcds = (obs_pcds-self.pcds_min)/(self.pcds_max-self.pcds_min)
        obs_pcds=obs_pcds*2-1
        return obs_pcds

    def get_obs_h2o(self,obj_state, hand_state):
        obj_pcds =self.obj_pcd_transforms_from_state(obj_state,self.obj_init_obj_pcds) #(B,2048,3)
        self.hand_model_forward(hand_state)
        hand_joints = self.hand_model.get_penetraion_keypoints() #(B,21,3)

        h2o_vec,h2o_dist =compute_h2o_minimum_vec(hand_joints, obj_pcds)

        h2o_vec = (h2o_vec-self.h2o_min)/(self.h2o_max-self.h2o_min)
        h2o_vec=h2o_vec*2-1
        return h2o_vec

    def get_seq_hand_pcd(self,hand_state, select_idxs=None, add_init_bias=True):
        if select_idxs is not None:
            hand_state = hand_state[select_idxs]
        B,T,D = hand_state.size()

        if add_init_bias:
            hand_state[...,2]+=self.init_hand_pos_z

        hand_state = hand_state.reshape(B*T, -1)#(B*T, 28)
        self.hand_model_forward(hand_state)
        pcd,_ = self.hand_model.get_surface_points() #(B,T,N,3)
        return pcd.reshape(B,T,-1,3)

    def detect_obj_pose_change(self, dist_thres=0.005):
        obj_state = self.get_object_state()
        distances = torch.norm(obj_state[:,:3]-self.init_obj_pos, dim=1)
        is_change_mask = (distances >= dist_thres).float()

        return is_change_mask


    def detect_h2o_close_enough(self,dist_thres=0.01):
        h2o_dist, obj_pcds = self.cal_h2o_distance()
        big_tips_dist = h2o_dist[:,-1] #(B,)

        #["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal", "robot0:thdistal"]
        big_tip_pos = self.fingertip_pos[:,-1,:] # (B,5,3)
        _, big_tips_dist2 = compute_h2o_minimum_vec(big_tip_pos, obj_pcds) #(B,)

        is_close_mask = (big_tips_dist <= dist_thres).float()
        return is_close_mask


    def cal_h2o_distance(self):
        hand_state = self.get_hand_state().unsqueeze(1)#（B，1，28）
        obj_state = self.get_object_state().unsqueeze(1) #（B，1，7）

        h2o_dist,obj_pcds = self.get_h2o_vector(hand_state, obj_state,return_dist=True) #(B,1, 21) (B,1,2048,3)
        return h2o_dist.squeeze(1), obj_pcds.squeeze(1)


    def apply_fingers_grip(self,actions, delta_angle=0.4,weight=1.5, tips_only=True):
        actions = actions.clone()

        if tips_only:
            actions[...,self.tips_idxs]+=delta_angle
            actions[...,self.big_tips_idx]-=delta_angle
        else:
            actions[...,6:]*=weight

        return actions


    # def get_pre_target_actions(self, iter):
    #     is_change_mask = self.detect_obj_pose_change()
    #     is_close_mask = self.detect_h2o_close_enough()
    #     check_mask = is_close_mask & is_change_mask
    #     self.check_mask+=check_mask
    #     check_mask = self.check_mask==1 #(判断第一次出现的帧，作为pre_target_actions)
    #
    #     self.iter_check_store[check_mask] = iter
    #     pre_target_actions = self.apply_fingers_grip(self.actions[check_mask])
    #
    #
    #     self.pre_target_actions[check_mask] = pre_target_actions
    #
    #
    #     actions = self.actions.clone()
    #
    #     achieve_target_mask = self.check_mask>=1
    #     actions[achieve_target_mask] = self.pre_target_actions[achieve_target_mask]
    #
    #     add_lift = 0.02*(iter - self.iter_check_store[achieve_target_mask]) #(N_check,)
    #     actions[achieve_target_mask,2]+=add_lift
    #
    #     return actions

    def get_pre_target_actions(self,actions, iter,previous_actions=None):
        is_change_mask = self.detect_obj_pose_change()
        is_close_mask = self.detect_h2o_close_enough()
        check_mask = is_close_mask.bool() | is_change_mask.bool()
        self.check_mask += check_mask
        check_mask = self.check_mask == 1

        self.iter_check_store[check_mask] = iter

        if previous_actions is not None:
            move_vectors = actions[:,:3]-previous_actions[:,:3] #(B,3)
            norms = move_vectors.norm(p=2,dim=-1,keepdim=True)
            move_vectors = move_vectors/(norms+1e-8)

            pre_target_actions = self.apply_fingers_grip(actions[check_mask],move_vectors[check_mask])
        else:
            pre_target_actions = self.apply_fingers_grip(actions[check_mask])

        self.pre_target_actions[check_mask] = pre_target_actions

        # actions = self.actions.clone()

        achieve_target_mask = self.check_mask >= 1
        actions[achieve_target_mask] = self.pre_target_actions[achieve_target_mask]

        add_lift = 0.02 * (iter - self.iter_check_store[achieve_target_mask])# (N_check,)


        actions[achieve_target_mask, 2] += add_lift

        return actions

    def hand_model_forward(self,hand_state):
        """
        hand_state (N, 28)
        """
        hand_pos = hand_state[...,:3].reshape(-1, 3)#(B*T, 3)
        hand_rot6d = euler_angles_to_matrix(hand_state[...,3:6],convention='XYZ').transpose(1,2).reshape(-1,9)[:,:6]#(B*T,6)
        hand_pose = torch.cat([hand_pos,hand_rot6d,hand_state[...,6:]],dim=1)
        self.hand_model.set_parameters(hand_pose)

    def get_seq_hand_mesh(self, hand_state,select_idxs=None, color='pink', return_id=0):
        """
        hand_state: (B,T,28)
        """

        if select_idxs is not None:
            hand_state = hand_state[select_idxs]
        B,T,D = hand_state.size()

        hand_state = hand_state.reshape(B*T, -1)#(B*T, 28)
        self.hand_model_forward(hand_state)


        hand_seq_mesh_list = []
        list_i = []
        for i in range(B*T):
            hand_mesh = self.hand_model.get_trimesh_data(i,color=color)
            hand_mesh = trimesh.util.concatenate(hand_mesh)
            if i%T==0:
                list_i=[]
                list_i.append(hand_mesh)
            else:
                list_i.append(hand_mesh)

            if i%T==T-1:
                hand_seq_mesh_list.append(list_i)



        return hand_seq_mesh_list


    def get_seq_hand_pcd(self,hand_state, select_idxs=None, add_init_bias=True, return_joints=True):
        if select_idxs is not None:
            hand_state = hand_state[select_idxs]
        B,T,D = hand_state.size()

        if add_init_bias:
            hand_state[...,2]+=self.init_hand_pos_z

        hand_state = hand_state.reshape(B*T, -1)#(B*T, 28)
        self.hand_model_forward(hand_state)
        pcd,_ = self.hand_model.get_surface_points() #(B*T,N,3)

        if return_joints:
            hand_joints = self.hand_model.get_penetraion_keypoints() #(B*T,21,3)
            return pcd.reshape(B,T,-1,3), hand_joints.reshape(B, T, -1, 3)
        return pcd.reshape(B,T,-1,3), None


    def get_hand_joints(self, hand_state, select_idxs=None, add_init_bias=True):
        if select_idxs is not None:
            hand_state = hand_state[select_idxs]
        B,T,D = hand_state.size()

        if add_init_bias:
            hand_state[...,2]+=self.init_hand_pos_z

        hand_state = hand_state.reshape(B*T, -1)#(B*T, 28)
        self.hand_model_forward(hand_state)
        hand_joints = self.hand_model.get_penetraion_keypoints()  # (B*T,21,3)
        return hand_joints.reshape(B,T,-1,3)

    def get_h2o_vector(self, hand_state, obj_state, select_idxs=None, add_init_bias=True, return_dist=False):

        device = obj_state.device
        obj_scale = torch.tensor(self.obj_trajs_info['obj_scale'], device=device).unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
        obj_rotmat = torch.tensor(self.obj_trajs_info['obj_rotmat'], device=device)  # (B,3,3)
        obj_idxs = torch.tensor(self.object_idxs, device=device)
        if select_idxs is not None:
            obj_state = obj_state[select_idxs]
            obj_scale = obj_scale[select_idxs] # (B,1,1)
            obj_rotmat = obj_rotmat[select_idxs]  # (B,3,3)
            obj_idxs = self.object_idxs[select_idxs]

        B,T,D = obj_state.size()
        if len(self.object_code_list) == 1:
            obj_pcds = torch.tensor(self.obj_sample_points,dtype=torch.float32).unsqueeze(0).repeat(B,1,1) #(B, 2048, 3)
        else:
            obj_pcd_list = [torch.tensor(self.obj_sample_points_list[idx], dtype=torch.float32, device=device) for idx in obj_idxs]
            obj_pcds = torch.stack(obj_pcd_list, dim=0)
        obj_pcds = torch.bmm(obj_pcds, obj_rotmat.transpose(1,2))*obj_scale  #(B, 2048, 3)
        obj_pcds = self.obj_pcd_transforms_from_state(obj_state, obj_pcds) #(B,T,  2048, 3)

        hand_joints = self.get_hand_joints(hand_state.cpu(), select_idxs, add_init_bias=add_init_bias).to(device) #(B,T,N,3)


        h2o_vec,h2o_dist =compute_h2o_minimum_vec(hand_joints.reshape(B*T,-1,3), obj_pcds.reshape(B*T,-1,3))
        if return_dist:
            return h2o_dist.reshape(B, T, -1), obj_pcds

        return h2o_vec.reshape(B,T,-1,3)



    def html_save(self, obj_seq_state, hand_seq_state,success_idx=None,key_str='', add_init_bias=True,extra_hand_state=None):
        """
        obj_seq_state(B,T,7)
        hand_seq_state(B,T,28)
        """
        _,T, _ = obj_seq_state.size()

        if add_init_bias:
            hand_seq_state[...,2]+=self.init_hand_pos_z

        obj_seq_meshes_list = self.get_seq_object_mesh(obj_seq_state, success_idx) #list (B,T)
        hand_seq_mesh_list = self.get_seq_hand_mesh(hand_seq_state, success_idx)  #list (B,T)
        B = len(obj_seq_meshes_list)

        extra_hand_seq_mesh_list=None
        if extra_hand_state is not None:
            extra_hand_state[...,2]+=self.init_hand_pos_z
            extra_hand_seq_mesh_list = self.get_seq_hand_mesh(extra_hand_state, success_idx,color='red')

        for i in range(B):
            idx = success_idx[i]
            obj_seq_mesh_i, hand_seq_mesh_i = obj_seq_meshes_list[i],  hand_seq_mesh_list[i]
            if isinstance(extra_hand_seq_mesh_list,list):
                extra_seq_meshes_list_i = extra_hand_seq_mesh_list[i]

            name = self.object_code_list[0]+'_seq{}_'.format(idx)+key_str
            if extra_hand_seq_mesh_list is not None:
                html_antmation_save(obj_seq_mesh_i, hand_seq_mesh_i, extra_seq_meshes_list_i, name=name)
            else:
                html_antmation_save(obj_seq_mesh_i, hand_seq_mesh_i, name=name)

            a=1


    def get_object_state(self):
        obj_pos = self.root_state_tensor.view(self.num_envs, -1, 13)[:, self.objects[0], :3]  # (B,3)
        obj_rot = self.root_state_tensor.view(self.num_envs, -1, 13)[:, self.objects[0], 3:7]  # (B,4)

        return torch.cat([obj_pos, obj_rot], dim=1) #(B, 7)

    def get_hand_state(self, add_init_bias=True):
        # hand_pose_params= []
        # for i in range(self.num_envs):
        #     hand_pose_params.append(self.gym.get_actor_dof_states(self.envs[i], self.o6_hands[i], gymapi.STATE_POS)['pos'])# (28,)
        # hand_pose_params = np.stack(hand_pose_params,axis=0)
        hand_pose_params = self.o6_hand_dof_pos.clone()
        if add_init_bias:
            hand_pose_params[:, 2] += self.init_hand_pos_z

        return hand_pose_params

    def get_object_sample_points(self, obj_mesh, point_num=2048):
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(obj_mesh.vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(obj_mesh.faces)
        pcd = mesh_o3d.sample_points_poisson_disk(point_num)
        pcd = np.asarray(pcd.points)
        return pcd

    def get_obj_half_height(self,object_scale, object_rotmat, object_id=0):
        obj_sample_points = self.obj_sample_points_list[object_id]

        pcd = np.matmul(obj_sample_points, object_rotmat.T)* object_scale
        min_z = np.min(pcd[:, 2])
        obj_half_height = min_z #* object_scale
        return obj_half_height, pcd
    def set_obj_start_root_state(self, obj_half_height,object_rotmat):
        r = R.from_matrix(object_rotmat)
        rot_quat = r.as_quat()

        object_z = self.table_height - obj_half_height+0.005
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.0, 0.0, object_z)  # gymapi.Vec3(0.0, 0.0, 0.72)
        # object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)
        # object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)
        object_start_pose.r  = gymapi.Quat(rot_quat[0], rot_quat[1], rot_quat[2], rot_quat[3])
        return object_start_pose

    def _load_object(self, env_ptr, goal_start_pose, i, object_asset_dict, object_idx_this_env, object_start_pose,
                     scale):
        object_handle = self.gym.create_actor(env_ptr, object_asset_dict[object_idx_this_env], object_start_pose,
                                              "object", i, 0, 0)
        self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                       object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z,
                                       object_start_pose.r.w,
                                       0, 0, 0, 0, 0, 0])
        self.goal_init_state.append([goal_start_pose.p.x, goal_start_pose.p.y, goal_start_pose.p.z,
                                     goal_start_pose.r.x, goal_start_pose.r.y, goal_start_pose.r.z,
                                     goal_start_pose.r.w,
                                     0, 0, 0, 0, 0, 0])
        object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
        self.object_indices.append(object_idx)
        self.gym.set_actor_scale(env_ptr, object_handle, scale)
        return object_handle

    def _load_o6_hand(self, env_ptr, i, o6_hand_asset, o6_hand_dof_props, o6_hand_start_pose):
        o6_hand_actor = self.gym.create_actor(env_ptr, o6_hand_asset, o6_hand_start_pose, "hand", i, -1, 0)
        self.hand_start_states.append(
            [o6_hand_start_pose.p.x, o6_hand_start_pose.p.y, o6_hand_start_pose.p.z,
             o6_hand_start_pose.r.x, o6_hand_start_pose.r.y, o6_hand_start_pose.r.z,
             o6_hand_start_pose.r.w,
             0, 0, 0, 0, 0, 0])
        self.gym.set_actor_dof_properties(env_ptr, o6_hand_actor, o6_hand_dof_props)
        hand_idx = self.gym.get_actor_index(env_ptr, o6_hand_actor, gymapi.DOMAIN_SIM)
        self.hand_indices.append(hand_idx)
        # randomize colors and textures for rigid body
        num_bodies = self.gym.get_actor_rigid_body_count(env_ptr, o6_hand_actor)
        hand_color = [147 / 255, 215 / 255, 160 / 255]
        hand_rigid_body_index = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19, 20],
                                 [21, 22, 23, 24, 25]]
        for n in self.agent_index[0]:
            for m in n:
                for o in hand_rigid_body_index[m]:
                    self.gym.set_rigid_body_color(env_ptr, o6_hand_actor, o, gymapi.MESH_VISUAL,
                                                  gymapi.Vec3(*hand_color))
        # create fingertip force-torque sensors
        # if self.obs_type == "full_state" or self.asymmetric_obs:
        self.gym.enable_actor_dof_force_sensors(env_ptr, o6_hand_actor)
        return o6_hand_actor

    def _load_table_asset(self):
        table_dims = gymapi.Vec3(1, 1, self.table_height)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, gymapi.AssetOptions())
        return table_asset, table_dims


    def get_object_name(self, object_code):
        object_code = object_code.split('.')[0]
        if 'ddg-gd' in object_code or 'ddg-kit' in object_code:
            object_name = object_code.split('_')[1]
        elif 'ddg-ycb' in object_code or "mujoco" in object_code:
            object_name = object_code.split('_')[-1]
        else:
            object_name = object_code.split('-')[1]

        object_name = object_name.lower()

        return object_name

    def _load_object_asset(self, assets_path):
        object_asset_dict = {}
        goal_asset_dict = {}
        self.num_object_bodies_list = []
        self.num_object_shapes_list = []
        # mesh_path = osp.join(assets_path, 'meshdatav3_scaled')
        self.asset_root = self.cfg["env"]["asset"]["assetRoot"]
        self.obj_asset_root = self.asset_root + self.cfg["env"]["asset"]["assetFileNameObj"]
        self.raw_obj_asset_root = self.asset_root + self.cfg["env"]["asset"]["assetFileNameObj_raw"]
        for object_id, object_code in enumerate(self.object_code_list):
            # load manipulated object and goal assets
            object_asset_options = gymapi.AssetOptions()
            if self.cfg["env"]["set_obj_mass"]==False:
                object_asset_options.density = 1
            object_asset_options.fix_base_link = False
            # object_asset_options.disable_gravity = True
            object_asset_options.override_com = True
            object_asset_options.override_inertia = True
            object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            object_asset = None
            object_asset_file = "coacd_1.urdf"
            if self.cfg['env']['obj_type'] in {'seen', 'one', 'unseen'}:
                object_asset = self.gym.load_asset(
                    self.sim,
                    self.obj_asset_root + f'{object_code}' + "/coacd",
                    object_asset_file,
                    object_asset_options,
                )
            if object_asset is None:
                print(object_code)
            assert object_asset is not None

            object_asset_options.disable_gravity = True
            goal_asset = self.gym.create_sphere(self.sim, 0.005, object_asset_options)

            # self.num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
            # self.num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)
            self.num_object_bodies_list.append(self.gym.get_asset_rigid_body_count(object_asset))
            self.num_object_shapes_list.append(self.gym.get_asset_rigid_shape_count(object_asset))
            # set object dof properties
            self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)
            object_dof_props = self.gym.get_asset_dof_properties(object_asset)
            self.object_dof_lower_limits = []
            self.object_dof_upper_limits = []

            for i in range(self.num_object_dofs):
                self.object_dof_lower_limits.append(object_dof_props['lower'][i])
                self.object_dof_upper_limits.append(object_dof_props['upper'][i])

            self.object_dof_lower_limits = to_torch(self.object_dof_lower_limits, device=self.device)
            self.object_dof_upper_limits = to_torch(self.object_dof_upper_limits, device=self.device)
            object_asset_dict[object_id] = object_asset
            goal_asset_dict[object_id] = goal_asset
        return goal_asset_dict, object_asset_dict

    def _load_o6_hand_asset(self):
        asset_root = "../../assets"
        o6_hand_asset_file = "linkerhand/o6/right/linkerhand_o6_right6d.urdf"
        table_texture_files = "../assets/textures/texture_wood_brown_1033760.jpg"
        table_texture_handle = self.gym.create_texture_from_file(self.sim, table_texture_files)
        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            o6_hand_asset_file = self.cfg["env"]["asset"].get("assetFileName", o6_hand_asset_file)
        # load o6 hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 1
        asset_options.linear_damping = 50
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        # o6_hand_asset = self.gym.load_asset(self.sim, asset_root, o6_hand_asset_file, asset_options)
        o6_hand_asset = self.gym.load_asset(self.sim, asset_root, o6_hand_asset_file, asset_options)
        self.num_o6_hand_bodies = self.gym.get_asset_rigid_body_count(o6_hand_asset)
        self.num_o6_hand_shapes = self.gym.get_asset_rigid_shape_count(o6_hand_asset)
        self.num_o6_hand_dofs = self.gym.get_asset_dof_count(o6_hand_asset)
        self.num_o6_hand_actuators = self.gym.get_asset_actuator_count(o6_hand_asset)
        self.num_o6_hand_tendons = self.gym.get_asset_tendon_count(o6_hand_asset)
        print("self.num_o6_hand_bodies: ", self.num_o6_hand_bodies)
        print("self.num_o6_hand_shapes: ", self.num_o6_hand_shapes)
        print("self.num_o6_hand_dofs: ", self.num_o6_hand_dofs)
        print("self.num_o6_hand_actuators: ", self.num_o6_hand_actuators)
        print("self.num_o6_hand_tendons: ", self.num_o6_hand_tendons)
        # tendon set up

        # limit_stiffness = 10
        # t_damping = 5
        #
        # relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        # tendon_props = self.gym.get_asset_tendon_properties(o6_hand_asset)
        # for i in range(self.num_o6_hand_tendons):
        #     for rt in relevant_tendons:
        #         if self.gym.get_asset_tendon_name(o6_hand_asset, i) == rt:
        #             tendon_props[i].limit_stiffness = limit_stiffness
        #             tendon_props[i].damping = t_damping
        # self.gym.set_asset_tendon_properties(o6_hand_asset, tendon_props)
        # actuated_dof_names = [self.gym.get_asset_actuator_joint_name(o6_hand_asset, i) for i in
        #                       range(self.num_o6_hand_actuators)]
        # self.actuated_dof_indices = [self.gym.find_asset_dof_index(o6_hand_asset, name) for name in
        #                              actuated_dof_names]
        # set o6 hand dof properties
        # o6_hand_dof_props = self.gym.get_asset_dof_properties(o6_hand_asset)
        # o6_hand_dof_props['damping'][6:]*=self.cfg['env']['damping_w']
        # o6_hand_dof_props['stiffness'][6:]*=self.cfg['env']['stiffness_w']
        self.o6_dof_names = [self.gym.get_asset_dof_name(o6_hand_asset, i) for i in range(self.num_o6_hand_dofs)]
        self.dof_dict = {name: i for i, name in enumerate(self.o6_dof_names)}

        o6_dof_props = self.gym.get_asset_dof_properties(o6_hand_asset)
        o6_dof_props['driveMode'].fill(gymapi.DOF_MODE_POS)

        virtual_joints = [
            "virtual_joint_x", "virtual_joint_y", "virtual_joint_z",
            "virtual_joint_roll", "virtual_joint_pitch", "virtual_joint_yaw"
        ]
        for joint_name in virtual_joints:
            if joint_name in self.dof_dict:
                idx = self.dof_dict[joint_name]
                o6_dof_props['stiffness'][idx] = float(self.cfg['env'].get('wrist_stiffness', 200000.0))
                o6_dof_props['damping'][idx] = float(self.cfg['env'].get('wrist_damping', 500.0))
                o6_dof_props['effort'][idx] = float(self.cfg['env'].get('wrist_effort', 1000.0))

        linkerhand_joints = [
            "rh_thumb_cmc_yaw", "rh_thumb_cmc_pitch", "rh_thumb_ip",
            "rh_index_mcp_pitch", "rh_index_dip",
            "rh_middle_mcp_pitch", "rh_middle_dip",
            "rh_ring_mcp_pitch", "rh_ring_dip",
            "rh_pinky_mcp_pitch", "rh_pinky_dip"
        ]
        finger_stiffness = float(self.cfg['env'].get('finger_stiffness', 120.0))
        finger_damping = float(self.cfg['env'].get('finger_damping', 6.0))
        finger_effort = float(self.cfg['env'].get('finger_effort', 200.0))
        for joint_name in linkerhand_joints:
            if joint_name in self.dof_dict:
                idx = self.dof_dict[joint_name]
                o6_dof_props['stiffness'][idx] = finger_stiffness
                o6_dof_props['damping'][idx] = finger_damping
                o6_dof_props['effort'][idx] = finger_effort
            else:
                print(f"[警告] URDF 中未找到手指关节: {joint_name}")


        # =====================================================================
        # 3. 动态确定 6 个主动手指关节在整个 DoF 数组中的准确索引
        # =====================================================================
        self.actuated_dof_indices = [
            self.dof_dict["rh_thumb_cmc_yaw"],
            self.dof_dict["rh_thumb_cmc_pitch"],
            self.dof_dict["rh_index_mcp_pitch"],
            self.dof_dict["rh_middle_mcp_pitch"],
            self.dof_dict["rh_ring_mcp_pitch"],
            self.dof_dict["rh_pinky_mcp_pitch"]
        ]
        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.virtual_dof_indices = to_torch(
            [
                self.dof_dict["virtual_joint_x"],
                self.dof_dict["virtual_joint_y"],
                self.dof_dict["virtual_joint_z"],
                self.dof_dict["virtual_joint_roll"],
                self.dof_dict["virtual_joint_pitch"],
                self.dof_dict["virtual_joint_yaw"],
            ],
            dtype=torch.long,
            device=self.device,
        )

        # o6_dof_props = self.gym.get_asset_dof_properties(o6_hand_asset)
        # for i in range(self.num_o6_hand_dofs):
        #     o6_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
        #     o6_dof_props['stiffness'][i] *=self.cfg['env']['damping_w']
        #     o6_dof_props['damping'][i] *=self.cfg['env']['stiffness_w']

        self.o6_hand_dof_lower_limits = []
        self.o6_hand_dof_upper_limits = []
        self.o6_hand_dof_default_pos = []
        self.o6_hand_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()
        for i in range(self.num_o6_hand_dofs):
            self.o6_hand_dof_lower_limits.append(o6_dof_props['lower'][i])
            self.o6_hand_dof_upper_limits.append(o6_dof_props['upper'][i])
            self.o6_hand_dof_default_pos.append(0.0)
            self.o6_hand_dof_default_vel.append(0.0)
        virtual_joint_indices = [
            self.dof_dict["virtual_joint_x"], self.dof_dict["virtual_joint_y"], self.dof_dict["virtual_joint_z"],
            self.dof_dict["virtual_joint_roll"], self.dof_dict["virtual_joint_pitch"],
            self.dof_dict["virtual_joint_yaw"]
        ]
        for v_idx in virtual_joint_indices:
            self.o6_hand_dof_lower_limits[v_idx] = -10.0
            self.o6_hand_dof_upper_limits[v_idx] = 10.0

        self.o6_hand_dof_lower_limits = to_torch(self.o6_hand_dof_lower_limits, device=self.device)
        self.o6_hand_dof_upper_limits = to_torch(self.o6_hand_dof_upper_limits, device=self.device)
        self.o6_hand_dof_default_pos = to_torch(self.o6_hand_dof_default_pos, device=self.device)
        self.o6_hand_dof_default_vel = to_torch(self.o6_hand_dof_default_vel, device=self.device)
        return o6_hand_asset, o6_dof_props, table_texture_handle

    def o6_actions_to_dof_targets(self, actions, wrist_z_offset=0.0, wrist_xy_away_offset=0.0):
        wrist_actions = actions[:, 0:6].clone()
        active_fingers = actions[:, 6:12]
        if wrist_z_offset != 0.0:
            wrist_actions[:, 2] += float(wrist_z_offset)
        if wrist_xy_away_offset != 0.0:
            xy = wrist_actions[:, 0:2]
            xy_norm = torch.norm(xy, dim=-1, keepdim=True).clamp_min(1e-6)
            wrist_actions[:, 0:2] += xy / xy_norm * float(wrist_xy_away_offset)

        if self.actions_are_normalized:
            active_lower = self.o6_hand_dof_lower_limits[self.actuated_dof_indices]
            active_upper = self.o6_hand_dof_upper_limits[self.actuated_dof_indices]
            fingers_physical = scale(active_fingers, active_lower, active_upper)
        else:
            fingers_physical = active_fingers

        full_targets = torch.zeros((actions.shape[0], self.num_o6_hand_dofs), device=self.device)

        virtual_joints = ["virtual_joint_x", "virtual_joint_y", "virtual_joint_z",
                          "virtual_joint_roll", "virtual_joint_pitch", "virtual_joint_yaw"]
        for i, v_joint in enumerate(virtual_joints):
            if v_joint in self.dof_dict:
                full_targets[:, self.dof_dict[v_joint]] = wrist_actions[:, i]

        full_targets[:, self.dof_dict["rh_thumb_cmc_yaw"]] = fingers_physical[:, 0]
        full_targets[:, self.dof_dict["rh_thumb_cmc_pitch"]] = fingers_physical[:, 1]
        full_targets[:, self.dof_dict["rh_thumb_ip"]] = fingers_physical[:, 1] * 1.86

        full_targets[:, self.dof_dict["rh_index_mcp_pitch"]] = fingers_physical[:, 2]
        full_targets[:, self.dof_dict["rh_index_dip"]] = fingers_physical[:, 2] * 0.89

        full_targets[:, self.dof_dict["rh_middle_mcp_pitch"]] = fingers_physical[:, 3]
        full_targets[:, self.dof_dict["rh_middle_dip"]] = fingers_physical[:, 3] * 0.89

        full_targets[:, self.dof_dict["rh_ring_mcp_pitch"]] = fingers_physical[:, 4]
        full_targets[:, self.dof_dict["rh_ring_dip"]] = fingers_physical[:, 4] * 0.89

        full_targets[:, self.dof_dict["rh_pinky_mcp_pitch"]] = fingers_physical[:, 5]
        full_targets[:, self.dof_dict["rh_pinky_dip"]] = fingers_physical[:, 5] * 0.89

        return tensor_clamp(
            full_targets,
            self.o6_hand_dof_lower_limits,
            self.o6_hand_dof_upper_limits,
        )

    def initial_wrist_z_offset(self, step_id):
        offset = float(self.cfg['env'].get('o6_initial_wrist_z_offset', 0.0))
        decay_steps = int(self.cfg['env'].get('o6_initial_wrist_z_offset_decay_steps', 0))
        if offset == 0.0:
            return 0.0
        if step_id < 0:
            return offset
        if decay_steps <= 0:
            return 0.0
        alpha = max(0.0, 1.0 - float(step_id - 1) / float(decay_steps))
        return offset * alpha

    def wrist_z_offset(self, step_id):
        base_offset = float(self.cfg['env'].get('o6_wrist_z_offset', 0.0))
        return base_offset + self.initial_wrist_z_offset(step_id)

    def initial_wrist_xy_away_offset(self, step_id):
        offset = float(self.cfg['env'].get('o6_initial_wrist_xy_away_offset', 0.0))
        decay_steps = int(self.cfg['env'].get('o6_initial_wrist_xy_away_offset_decay_steps', 0))
        if offset == 0.0:
            return 0.0
        if step_id < 0:
            return offset
        if decay_steps <= 0:
            return 0.0
        alpha = max(0.0, 1.0 - float(step_id - 1) / float(decay_steps))
        return offset * alpha

    def clean_sim(self):
        if getattr(self, "_sim_destroyed", False):
            return
        self.camera_rgb_tensor_list=[]

        self.num_object_bodies_list=[]
        self.num_object_shapes_list = []
        self.object_code_list = []
        if self.headless == False:
            # self.gym.destroy_viewer(self.my_viewer)
            if self.viewer is not None:
                self.gym.destroy_viewer(self.viewer)


        self.gym.destroy_sim(self.sim)
        self._sim_destroyed = True


    def compute_reward(self, actions, id=-1):
        self.dof_pos = self.o6_hand_dof_pos
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.current_successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.object_init_z,
            self.id, self.object_id_buf, self.dof_pos, self.rew_buf, self.reset_buf, self.reset_goal_buf,
            self.progress_buf, self.successes, self.current_successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_handle_pos, self.object_back_pos, self.object_rot,
            self.goal_pos, self.goal_rot,
            self.o6_palm_pos, self.o6_thumb_tip_pos, self.o6_index_tip_pos, self.o6_middle_tip_pos,
            self.o6_ring_tip_pos, self.o6_pinky_tip_pos,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor,self.goal_cond
        )

        self.extras['successes'] = self.successes
        self.extras['current_successes'] = self.current_successes
        self.extras['consecutive_successes'] = self.consecutive_successes

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(
                direct_average_successes / (self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(
                    self.total_successes / self.total_resets))


    def sim_refresh(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

    def compute_observations_o6(self):
        # TODO:using dexrep
        self.sim_refresh()

        # if self.obs_type == "full_state" or self.asymmetric_obs:
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_handle_pos = self.object_pos  ##+ quat_apply(self.object_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.06)
        self.object_back_pos = self.object_pos + quat_apply(self.object_rot,to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.04)
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        idx = self.hand_body_idx_dict['palm']
        self.o6_palm_pos = self.rigid_body_states[:, idx, 0:3]
        self.o6_palm_rot = self.rigid_body_states[:, idx, 3:7]
        self.o6_palm_pos = self.o6_palm_pos + quat_apply(self.o6_palm_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.o6_palm_pos = self.o6_palm_pos + quat_apply(self.o6_palm_rot,to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        # right hand finger
        if self.use_dexrep or self.use_pnG or self.use_obs_pcds:
            self.dexrep_hand_state = self.rigid_body_states[:, self.dexrep_hand_indices, :].view(self.num_envs, -1, 13)
            self.dexrep_hand_pos = self.dexrep_hand_state[:, :, 0:3]
            self.dexrep_hand_vel = self.dexrep_hand_state[:, :, 7:13]
            # compute fingertip
            idx = 0
            self.o6_thumb_tip_pos, self.o6_thumb_tip_rot = self.dexrep_hand_state[:, idx, 0:3], self.dexrep_hand_state[:, idx, 3:7]
            self.o6_thumb_tip_pos = self.o6_thumb_tip_pos + quat_apply(self.o6_thumb_tip_rot,
                                                                         to_torch([0, 0, 1], device=self.device).repeat(
                                                                             self.num_envs, 1) * 0.02)

            idx = 1
            self.o6_index_tip_pos, self.o6_index_tip_rot = self.dexrep_hand_state[:, idx, 0:3], self.dexrep_hand_state[:, idx, 3:7]
            self.o6_index_tip_pos = self.o6_index_tip_pos + quat_apply(self.o6_index_tip_rot,
                                                                         to_torch([0, 0, 1], device=self.device).repeat(
                                                                             self.num_envs, 1) * 0.02)

            idx = 2
            self.o6_middle_tip_pos = self.dexrep_hand_state[:, idx, 0:3]
            self.o6_middle_tip_rot = self.dexrep_hand_state[:, idx, 3:7]
            self.o6_middle_tip_pos = self.o6_middle_tip_pos + quat_apply(self.o6_middle_tip_rot,
                                                                         to_torch([0, 0, 1], device=self.device).repeat(
                                                                             self.num_envs, 1) * 0.02)

            idx = 3
            self.o6_ring_tip_pos = self.dexrep_hand_state[:, idx, 0:3]
            self.o6_ring_tip_rot = self.dexrep_hand_state[:, idx, 3:7]
            self.o6_ring_tip_pos = self.o6_ring_tip_pos + quat_apply(self.o6_ring_tip_rot,
                                                                         to_torch([0, 0, 1], device=self.device).repeat(
                                                                             self.num_envs, 1) * 0.02)

            idx = 4
            self.o6_pinky_tip_pos = self.dexrep_hand_state[:, idx, 0:3]
            self.o6_pinky_tip_rot = self.dexrep_hand_state[:, idx, 3:7]
            self.o6_pinky_tip_pos = self.o6_pinky_tip_pos + quat_apply(self.o6_pinky_tip_rot,
                                                                         to_torch([0, 0, 1], device=self.device).repeat(
                                                                             self.num_envs, 1) * 0.02)
            # concatenate
            fingertip_pos = torch.cat(
                (self.o6_thumb_tip_pos.unsqueeze(-2),
                 self.o6_index_tip_pos.unsqueeze(-2),
                 self.o6_middle_tip_pos.unsqueeze(-2),
                 self.o6_ring_tip_pos.unsqueeze(-2),
                 self.o6_pinky_tip_pos.unsqueeze(-2)),
                dim=1
            )
            self.dexrep_hand_pos = torch.cat(    # expected [B, 20, 3]
                (fingertip_pos, self.dexrep_hand_pos),
                dim=1
            )
            if self.cfg["env"].get("debug_o6_dexrep_sites", False) and not hasattr(self, "_debug_o6_dexrep_sites_printed"):
                print("o6_dexrep_sites: dexrep_hand_state={}, joints_sate={}".format(
                    tuple(self.dexrep_hand_state.shape),
                    tuple(self.dexrep_hand_pos.shape),
                ))
                self._debug_o6_dexrep_sites_printed = True

        # self.fingertip_state = self.rigid_body_states[self.fingertip_indices].view(self.num_envs, -1, 13)
        # self.fingertip_pos = self.fingertip_state[:, :, 0:3]
        # self.fingertip_ori = self.fingertip_state[:, :, 3:7]
        # self.fingertip_lin_vel = self.fingertip_state[:, :, 7:10]
        # self.fingertip_ang_vel = self.fingertip_state[:, :, 10:13]
        # self.fingertip_vel = self.fingertip_state[:, :, 7:13]
        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        if int(self.cfg["env"]["obs_dim"].get("prop", 222)) == 77:
            base_state = self.compute_o6_compact_state()
        else:
            base_state = self.compute_full_state()
        base_state = torch.clamp(base_state, -self.cfg["env"]["clip_observations"],
                                 self.cfg["env"]["clip_observations"])

        if self.obs_type in ['DexRep']:
            assert self.use_dexrep
            dexrep_obs = self.DexRepEncoder.pre_observation(
                obj_pos=self.object_pos,
                obj_rot=self.object_rot,
                hand_pos=self.o6_palm_pos,
                hand_rot=self.o6_palm_rot,
                joints_sate=self.dexrep_hand_pos,
                clip_range=self.cfg["env"]["clip_observations"]
            )
            # dexrep_obs = torch.clamp(dexrep_obs, -self.cfg["env"]["clip_observations"],
            #                      self.cfg["env"]["clip_observations"])
            self.obs_buf = torch.cat(
                (base_state, dexrep_obs),
                dim=1
            )

        elif self.obs_type in ['obs_pcds']:
            hand_state = self.get_hand_state()
            obj_state = self.get_object_state()
            self.obs_buf = self.get_obs_pcds(obj_state, hand_state)  # (B,512,3)

            self.unactions = unscale(self.o6_hand_dof_pos, self.o6_hand_dof_lower_limits,
                                     self.o6_hand_dof_upper_limits)  # (B,28)

        elif self.obs_type in ['obs_h2o']:
            hand_state = self.get_hand_state()
            obj_state = self.get_object_state()
            self.obs_buf = self.get_obs_h2o(obj_state, hand_state)  # (B,21,3)
            self.unactions = unscale(self.o6_hand_dof_pos, self.o6_hand_dof_lower_limits,
                                     self.o6_hand_dof_upper_limits)  # (B,28)

        else:
            raise AttributeError(f'{self.obs_type} not include..')

    def compute_observations(self):
        if self.cfg["env"].get("o6_policy_obs_mode", "prev_action_obj_rot") == "dexrep":
            return self.compute_observations_o6()

        # 1. 刷新物理引擎状态缓冲
        self.sim_refresh()
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.object_handle_pos = self.object_pos
        self.object_back_pos = self.object_pos + quat_apply(self.object_rot,
                                                            to_torch([1, 0, 0], device=self.device).repeat(
                                                                self.num_envs, 1) * 0.04)

        self.o6_palm_pos = self.o6_hand_dof_pos[:, 0:3]
        self.o6_palm_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.o6_palm_rot[:, 3] = 1.0  # 默认四元数 w=1

        self.o6_thumb_tip_pos = self.o6_palm_pos
        self.o6_index_tip_pos = self.o6_palm_pos
        self.o6_middle_tip_pos = self.o6_palm_pos
        self.o6_ring_tip_pos = self.o6_palm_pos
        self.o6_pinky_tip_pos = self.o6_palm_pos

        dummy_obs = torch.zeros((self.num_envs, 21), device=self.device, dtype=torch.float32)

        hand_base_pos = self.o6_hand_dof_pos[:, 0:6].clone()
        # table_base_z = self.table_height + 0.005
        # hand_base_pos[:, 2] -= table_base_z

        hand_finger_pos = self.o6_hand_dof_pos[:, self.actuated_dof_indices]
        current_hand_12d = torch.cat([hand_base_pos, hand_finger_pos], dim=-1)
        dummy_obs[:, 0:12] = current_hand_12d

        obj_quat_xyzw = self.object_pose[:, 3:7]
        obj_quat_wxyz = torch.cat([obj_quat_xyzw[:, 3:4], obj_quat_xyzw[:, 0:3]], dim=-1)
        obj_rot_mat = quaternion_to_matrix(obj_quat_wxyz).reshape(self.num_envs, 9)
        dummy_obs[:, 12:21] = obj_rot_mat

        self.obs_buf[:] = dummy_obs

        if hasattr(self, 'states_buf'):
            self.states_buf[:] = dummy_obs

    def get_unpose_quat(self):
        if self.repose_z:
            self.unpose_z_theta_quat = quat_from_euler_xyz(
                torch.zeros_like(self.z_theta), torch.zeros_like(self.z_theta),
                -self.z_theta,
            )
        return

    def unpose_point(self, point):
        if self.repose_z:
            return self.unpose_vec(point)
            # return self.origin + self.unpose_vec(point - self.origin)
        return point

    def unpose_vec(self, vec):
        if self.repose_z:
            return quat_apply(self.unpose_z_theta_quat, vec)
        return vec

    def unpose_quat(self, quat):
        if self.repose_z:
            return quat_mul(self.unpose_z_theta_quat, quat)
        return quat

    def unpose_state(self, state):
        if self.repose_z:
            state = state.clone()
            state[:, 0:3] = self.unpose_point(state[:, 0:3])
            state[:, 3:7] = self.unpose_quat(state[:, 3:7])
            state[:, 7:10] = self.unpose_vec(state[:, 7:10])
            state[:, 10:13] = self.unpose_vec(state[:, 10:13])
        return state

    def get_pose_quat(self):
        if self.repose_z:
            self.pose_z_theta_quat = quat_from_euler_xyz(
                torch.zeros_like(self.z_theta), torch.zeros_like(self.z_theta),
                self.z_theta,
            )
        return

    def pose_vec(self, vec):
        if self.repose_z:
            return quat_apply(self.pose_z_theta_quat, vec)
        return vec

    def pose_point(self, point):
        if self.repose_z:
            return self.pose_vec(point)
            # return self.origin + self.pose_vec(point - self.origin)
        return point

    def pose_quat(self, quat):
        if self.repose_z:
            return quat_mul(self.pose_z_theta_quat, quat)
        return quat

    def pose_state(self, state):
        if self.repose_z:
            state = state.clone()
            state[:, 0:3] = self.pose_point(state[:, 0:3])
            state[:, 3:7] = self.pose_quat(state[:, 3:7])
            state[:, 7:10] = self.pose_vec(state[:, 7:10])
            state[:, 10:13] = self.pose_vec(state[:, 10:13])
        return state

    def compute_full_state(self, asymm_obs=False):

        self.get_unpose_quat()
        obs_buf = torch.zeros((self.num_envs, 222), device=self.device, dtype=torch.float)
        # unscale to (-1，1)
        num_ft_states = 13 * int(self.num_fingertips)  # 65 ##
        num_ft_force_torques = 6 * int(self.num_fingertips)  # 30 ##

        # 0：84
        obs_buf[:, 0:self.num_o6_hand_dofs] = unscale(self.o6_hand_dof_pos,
                                                               self.o6_hand_dof_lower_limits,
                                                               self.o6_hand_dof_upper_limits)
        obs_buf[:,self.num_o6_hand_dofs:2 * self.num_o6_hand_dofs] = self.vel_obs_scale * self.o6_hand_dof_vel
        obs_buf[:,2 * self.num_o6_hand_dofs:3 * self.num_o6_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor[:, :]
        fingertip_obs_start = 3 * self.num_o6_hand_dofs
        aux = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        for i in range(5):
            aux[:, i * 13:(i + 1) * 13] = self.unpose_state(aux[:, i * 13:(i + 1) * 13])
        # 84:149: ft states
        obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = aux

        # 149:179: ft sensors: do not need repose
        obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states + num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, :30]

        hand_pose_start = fingertip_obs_start + 95
        # 179:185: hand_pose
        obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.unpose_point(self.o6_palm_pos)
        euler_xyz = get_euler_xyz(self.unpose_quat(self.hand_orientations[self.hand_indices, :]))
        obs_buf[:, hand_pose_start + 3:hand_pose_start + 4] = euler_xyz[0].unsqueeze(-1)
        obs_buf[:, hand_pose_start + 4:hand_pose_start + 5] = euler_xyz[1].unsqueeze(-1)
        obs_buf[:, hand_pose_start + 5:hand_pose_start + 6] = euler_xyz[2].unsqueeze(-1)

        action_obs_start = hand_pose_start + 6
        # 185:209: action. O6 policy actions are 12-D; the remaining slots stay zero
        # to preserve the 222-D proprioceptive layout expected by DexRep models.
        aux = self.actions[:, :12]

        # aux[:, 0:3] = self.unpose_vec(aux[:, 0:3])
        # aux[:, 3:6] = self.unpose_vec(aux[:, 3:6])
        # obs_buf[:, action_obs_start:action_obs_start + 24] = aux
        if self.cfg['env']['env_mode']=='extract_obs' or self.id==-1:
            obs_buf[:, action_obs_start:action_obs_start + 12] = aux
        else:
            obs_buf[:, action_obs_start:action_obs_start + 12] = aux

        obj_obs_start = action_obs_start + 24  # 144
        # 209:222 object_pose, goal_pos
        obs_buf[:, obj_obs_start:obj_obs_start + 3] = self.unpose_point(self.object_pose[:, 0:3])
        obs_buf[:, obj_obs_start + 3:obj_obs_start + 7] = self.unpose_quat(self.object_pose[:, 3:7])
        obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.unpose_vec(self.object_linvel)
        obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.unpose_vec(self.object_angvel)
        # obs_buf[:, obj_obs_start + 13:obj_obs_start + 16] = self.unpose_vec(self.goal_pos - self.object_pos)

         # 207:236 goal
        # hand_goal_start = obj_obs_start + 16
        # obs_buf[:, hand_goal_start:hand_goal_start + 3] = self.delta_target_hand_pos
        # obs_buf[:, hand_goal_start + 3:hand_goal_start + 7] = self.delta_target_hand_rot
        # obs_buf[:, hand_goal_start + 7:hand_goal_start + 29] = self.delta_qpos

        # 236: visual feature
        # visual_feat_start = hand_goal_start + 29

        # 236: 300: visual feature
        # obs_buf[:, visual_feat_start:visual_feat_start + 64] = 0.1 * self.visual_feat_buf

        return obs_buf

    def compute_o6_compact_state(self):
        self.get_unpose_quat()
        obs_buf = torch.zeros((self.num_envs, 77), device=self.device, dtype=torch.float)

        obs_buf[:, 0:self.num_o6_hand_dofs] = unscale(
            self.o6_hand_dof_pos,
            self.o6_hand_dof_lower_limits,
            self.o6_hand_dof_upper_limits,
        )

        fingertip_state = self.fingertip_state.clone().reshape(self.num_envs, self.num_fingertips, 13)
        for i in range(self.num_fingertips):
            fingertip_state[:, i, :] = self.unpose_state(fingertip_state[:, i, :])
        obs_buf[:, 17:52] = fingertip_state[:, :, 0:7].reshape(self.num_envs, 35)

        obs_buf[:, 52:55] = self.unpose_point(self.o6_palm_pos)
        euler_xyz = get_euler_xyz(self.unpose_quat(self.hand_orientations[self.hand_indices, :]))
        obs_buf[:, 55:56] = euler_xyz[0].unsqueeze(-1)
        obs_buf[:, 56:57] = euler_xyz[1].unsqueeze(-1)
        obs_buf[:, 57:58] = euler_xyz[2].unsqueeze(-1)

        obs_buf[:, 58:70] = self.actions[:, :12]

        obs_buf[:, 70:73] = self.unpose_point(self.object_pose[:, 0:3])
        obs_buf[:, 73:77] = self.unpose_quat(self.object_pose[:, 3:7])
        return obs_buf

    def reset_target_pose(self, env_ids, apply_reset=False):

        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]

        # self.goal_states[env_ids, 3:7] = new_rot
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3]  # + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]

        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    def reset(self, env_ids, goal_env_ids):

        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_o6_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # reset o6 hand
        delta_max = self.o6_hand_dof_upper_limits - self.o6_hand_dof_default_pos
        delta_min = self.o6_hand_dof_lower_limits - self.o6_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5 + self.num_o6_hand_dofs]

        # pos = self.o6_hand_default_dof_pos  # + self.reset_dof_pos_noise * rand_delta

        first_action_12d = self.grasp_seqs[:, 0, :].clone()
        first_action_17d = self.o6_actions_to_dof_targets(
            first_action_12d,
            wrist_z_offset=self.wrist_z_offset(-1),
            wrist_xy_away_offset=self.initial_wrist_xy_away_offset(-1),
        )

        self.o6_hand_dof_pos[env_ids, :] = first_action_17d[env_ids, :]
        pos = first_action_17d[env_ids, :]

        self.o6_hand_dof_vel[env_ids, :] = self.o6_hand_dof_default_vel
        self.prev_targets[env_ids, :] = pos
        self.cur_targets[env_ids, :] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        all_hand_indices = torch.unique(torch.cat([hand_indices]).to(torch.int32))

        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        self.hand_linvels[hand_indices.to(torch.long), :] = 0
        self.hand_angvels[hand_indices.to(torch.long), :] = 0

        # Fixed-base virtual wrist DOF updates do not move rigid bodies until
        # PhysX advances. Sync the hand before resetting objects, otherwise the
        # first rollout frame can resolve contacts against a stale hand pose and
        # push env0's object away.
        if self.cfg.get("env", {}).get("infer_runtime", {}).get("sync_hand_before_object_reset", True):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.sim_refresh()

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = 0.0
        self.root_state_tensor[self.goal_object_indices[env_ids]] = self.goal_init_state[env_ids].clone()
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = 0.0

        all_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                              self.goal_object_indices[env_ids]]).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))
        if self.cfg.get("env", {}).get("infer_runtime", {}).get("refresh_after_reset", True):
            self.gym.fetch_results(self.sim, True)
            self.sim_refresh()

        if self.random_time:
            self.random_time = False
            self.progress_buf[env_ids] = torch.randint(0, self.max_episode_length, (len(env_ids),), device=self.device)
        else:
            self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

        return first_action_12d

    def direct_reset_to_first_frame(self):
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        goal_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        first_actions = self.reset(env_ids, goal_env_ids)
        self.actions = first_actions.clone().to(self.device)
        self.cur_targets[:] = self.o6_actions_to_dof_targets(
            self.actions,
            wrist_z_offset=self.wrist_z_offset(-1),
            wrist_xy_away_offset=self.initial_wrist_xy_away_offset(-1),
        )
        self.prev_targets[:] = self.cur_targets[:]
        all_hand_indices = torch.unique(torch.cat([self.hand_indices]).to(torch.int32))
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(all_hand_indices),
            len(all_hand_indices)
        )
        self.gym.fetch_results(self.sim, True)
        self.sim_refresh()
        self.compute_observations()
        self.rew_buf[:] = 0
        self.successes[:] = 0
        self.current_successes[:] = 0
        self.reset_buf[:] = 0
        self.progress_buf[:] = 0
        return self.obs_buf

    def pre_physics_step(self, actions, id):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        # if goals need reset in addition to other envspre_physics_step
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            first_actions = self.reset(env_ids, goal_env_ids)
            actions = first_actions

        self.get_pose_quat()
        # actions[:, 0:3] = self.pose_vec(actions[:, 0:3])
        # actions[:, 3:6] = self.pose_vec(actions[:, 3:6])
        self.actions = actions.clone().to(self.device)

        if self.cfg['env']['env_mode'] in {'bc_env_infer', 'extract_obs'}:
            next_targets = self.o6_actions_to_dof_targets(
                self.actions,
                wrist_z_offset=self.wrist_z_offset(id),
                wrist_xy_away_offset=self.initial_wrist_xy_away_offset(id),
            )
            if self.cfg['env']['env_mode'] == 'extract_obs' or self.cfg['env'].get('o6_control_wrist_each_step', False):
                self.cur_targets[:] = next_targets
                self.prev_targets[:] = self.cur_targets[:]
            else:
                self.cur_targets[:, self.actuated_dof_indices] = next_targets[:, self.actuated_dof_indices]
                self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        all_hand_indices = torch.unique(torch.cat([self.hand_indices]).to(torch.int32))
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.prev_targets),
            gymtorch.unwrap_tensor(all_hand_indices),
            len(all_hand_indices)
        )

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions, self.id)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                self.add_debug_lines(self.envs[i], self.object_pos[i], self.object_rot[i])
                # self.add_debug_lines(self.envs[i], self.object_back_pos[i], self.object_rot[i])
                # self.add_debug_lines(self.envs[i], self.goal_pos[i], self.object_rot[i])
                # self.add_debug_lines(self.envs[i], self.o6_palm_pos[i], self.o6_palm_rot[i])
                # self.add_debug_lines(self.envs[i], self.o6_thumb_tip_pos[i], self.o6_thumb_tip_rot[i])
                # self.add_debug_lines(self.envs[i], self.o6_index_tip_pos[i], self.o6_index_tip_rot[i])
                # self.add_debug_lines(self.envs[i], self.o6_middle_tip_pos[i], self.o6_middle_tip_rot[i])
                # self.add_debug_lines(self.envs[i], self.o6_ring_tip_pos[i], self.o6_ring_tip_rot[i])
                # self.add_debug_lines(self.envs[i], self.o6_pinky_tip_pos[i], self.o6_pinky_tip_rot[i])

                # self.add_debug_lines(self.envs[i], self.left_hand_ff_pos[i], self.o6_thumb_tip_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_mf_pos[i], self.o6_index_tip_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_rf_pos[i], self.o6_middle_tip_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_lf_pos[i], self.o6_ring_tip_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_th_pos[i], self.o6_pinky_tip_rot[i])

    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])


#####################################################################
###=========================jit functions=========================###
#####################################################################


# @torch.jit.script
def compute_hand_reward(
        object_init_z,
        id: int, object_id, dof_pos, rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, current_successes, consecutive_successes,
        max_episode_length: float, object_pos, object_handle_pos, object_back_pos, object_rot, target_pos, target_rot,
        o6_palm_pos, o6_thumb_tip_pos, o6_index_tip_pos, o6_middle_tip_pos, o6_ring_tip_pos, o6_pinky_tip_pos,
        dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
        actions, action_penalty_scale: float,
        success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
        fall_penalty: float, max_consecutive_successes: int, av_factor: float, goal_cond: bool
):
    # Distance from the hand to the object
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    reward = 0


    resets = reset_buf

    # Find out which envs hit the goal and update successes count
    # resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    goal_resets = resets
    # successes = torch.where(goal_dist <= 0.10, torch.ones_like(successes), successes)
    # successes = torch.where(object_pos[:, 2] >= target_pos[:, 2], torch.ones_like(successes), successes)


    # successes_ext =torch.where(object_pos[:,2]<=1.7, torch.ones_like(successes), torch.zeros_like(successes))
    successes_ext = torch.where(
        (object_pos[:, 0] >= -1.5) & (object_pos[:, 0] <= 1.5) &
        (object_pos[:, 1] >= -1.5) & (object_pos[:, 1] <= 1.5) &
        (object_pos[:, 2] < 2.0),
        torch.ones_like(successes),
        torch.zeros_like(successes)
    )
    successes =torch.where(goal_dist <= 0.12, torch.ones_like(successes),successes)
    successes = torch.where(object_pos[:, 2] >= target_pos[:, 2], torch.ones_like(successes), successes)
    successes = successes_ext * successes

    # successes = torch.where(object_pos[:, 2] <= target_pos[:, 2]+1, torch.ones_like(successes), successes)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    current_successes = torch.where(resets==1, successes, current_successes)
    cons_successes = torch.where(num_resets > 0, av_factor * finished_cons_successes / num_resets + (
                1.0 - av_factor) * consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, current_successes, cons_successes


def compute_hand_reward_rl(
        object_init_z,
        id: int, object_id, dof_pos, rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, current_successes, consecutive_successes,
        max_episode_length: float, object_pos, object_handle_pos, object_back_pos, object_rot, target_pos, target_rot,
        o6_palm_pos, o6_thumb_tip_pos, o6_index_tip_pos, o6_middle_tip_pos, o6_ring_tip_pos, o6_pinky_tip_pos,
        dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
        actions, action_penalty_scale: float,
        success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
        fall_penalty: float, max_consecutive_successes: int, av_factor: float, goal_cond: bool
):
    # Distance from the hand to the object
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    goal_o6_palm_dist = torch.norm(target_pos - o6_palm_pos, p=2, dim=-1)
    o6_palm_dist = torch.norm(object_handle_pos - o6_palm_pos, p=2, dim=-1)
    o6_palm_dist = torch.where(o6_palm_dist >= 0.5, 0.5 + 0 * o6_palm_dist, o6_palm_dist)

    o6_fingertip_dist = (torch.norm(object_handle_pos - o6_thumb_tip_pos, p=2, dim=-1) + torch.norm(
        object_handle_pos - o6_index_tip_pos, p=2, dim=-1)+ torch.norm(object_handle_pos - o6_middle_tip_pos, p=2, dim=-1) + torch.norm(
                object_handle_pos - o6_ring_tip_pos, p=2, dim=-1) + torch.norm(object_handle_pos - o6_pinky_tip_pos, p=2, dim=-1))
    o6_fingertip_dist = torch.where(o6_fingertip_dist >= 3.0, 3.0 + 0 * o6_fingertip_dist,o6_fingertip_dist)
    lowest = object_pos[:, 2]


    flag = (o6_fingertip_dist <= 0.6).int() + (o6_palm_dist <= 0.12).int()
    goal_o6_palm_rew = torch.zeros_like(o6_fingertip_dist)
    goal_o6_palm_rew = torch.where(flag == 2, 1 * (0.9 - 2 * goal_dist), goal_o6_palm_rew)

    hand_up = torch.zeros_like(o6_fingertip_dist)
    hand_up = torch.where(lowest >= 0.630, torch.where(flag == 2, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)
    hand_up = torch.where(lowest >= 0.80, torch.where(flag == 2, 0.2 - goal_o6_palm_dist * 0, hand_up), hand_up)

    flag = (o6_fingertip_dist <= 0.6).int() + (o6_palm_dist <= 0.12).int()
    bonus = torch.zeros_like(goal_dist)
    bonus = torch.where(flag == 2, torch.where(goal_dist <= 0.05, 1.0 / (1 + 10 * goal_dist), bonus), bonus)

    reward = -0.5 * o6_fingertip_dist - 1.0 * o6_palm_dist + goal_o6_palm_rew + hand_up + bonus


    resets = reset_buf

    # Find out which envs hit the goal and update successes count
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    goal_resets = resets
    successes = torch.where(goal_dist <= 0.05, torch.ones_like(successes), successes)
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    current_successes = torch.where(resets.bool(), successes, current_successes)
    cons_successes = torch.where(num_resets > 0, av_factor * finished_cons_successes / num_resets + (
                1.0 - av_factor) * consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, current_successes, cons_successes




@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot
