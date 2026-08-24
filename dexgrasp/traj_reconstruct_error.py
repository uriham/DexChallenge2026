import torch
import trimesh
import transforms3d
import numpy as np
import os.path as osp
from utils.hand_model import HandModel
from utils.vposer.vposer_model import VPoserShadow
from utils.rot6d import robust_compute_ortho6d_from_euler
from open3d import geometry as o3dg
from ActionDiffusion.utils.vis_utils import sp_animation

def split_and_stack_traj_list(traj_list, min_len=None):
    """
    traj_list: List of np.ndarray, each with shape (T_i, 28)

    Returns:
        traj_array: np.ndarray of shape (N, T_min, 28)
    """
    if min_len is not None:
        traj_list = [traj for traj in traj_list if traj.shape[0]>=min_len]

    T_min = min(traj.shape[0] for traj in traj_list)

    chunks = []
    for traj in traj_list:
        num_chunks = traj.shape[0] // T_min
        for i in range(num_chunks):
            chunk = traj[i * T_min: (i + 1) * T_min]  # shape (T_min, 28)
            chunks.append(chunk)

    traj_array = np.stack(chunks, axis=0)
    return traj_array


def html_antmation_save(hand_mesh_list, name='test'):

    FOR1 = o3dg.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    FOR1 = trimesh.Trimesh(vertices=np.asarray(FOR1.vertices),faces=np.asarray(FOR1.triangles)
                           ,vertex_colors=np.asarray(FOR1.vertex_colors))

    grasp_anim = sp_animation()
    for i,meshes_i in enumerate(hand_mesh_list):
            grasp_anim.add_frame([meshes_i, FOR1], ['hand', 'obj', 'XYZaxis'])


    grasp_anim.save_animation(osp.join('./', "{}.html".format(name)))
    a=1


class ShadowVposerReconstructor:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.dt = cfgs.dt
        self.device='cuda'
        self.use_gpos=cfgs.use_gpos
        self.data_path = cfgs.data_path
        self.obj_code_list = cfgs.obj_code_list
        self.vposer_model = VPoserShadow(latentD=5).to('cuda')
        self.vposer_model.load_state_dict(torch.load(cfgs.vposer_ckpt_path))

        self.data_load(cfgs.data_path, cfgs.seq_len)


        rotation = torch.tensor(transforms3d.euler.euler2mat(0, -np.pi / 3, 0, axes='rzxz'), dtype=torch.float, device=self.device)
        self.shadow_rot = rotation.T.ravel()[:6].unsqueeze(0)
        self.shadow_trans = torch.zeros(1, 3, device=self.device).to('cuda')

        model_base_path = "../assets/mjcf/"
        self.hand_model = HandModel(
            mjcf_path=model_base_path + 'shadow_hand_vis_new.xml', mesh_path=model_base_path + 'meshes',
            contact_points_path=model_base_path + 'contact_points.json',
            penetration_points_path=model_base_path + 'penetration_points.json',
            n_surface_points=512,
            device=self.device,
            use_joint21=True
        )
        if cfgs.html_save:
            self.html_save(self.data)

        a=1

    def html_save(self, poses):
        mesh_list = self.poses_to_meshes(poses)
        B=len(mesh_list)
        for i in range(B):
            hand_seq_mesh_i = mesh_list[i]
            html_antmation_save(hand_seq_mesh_i, name='hand_seq{}'.format(i))

    def data_load(self, data_path, seq_len=10):
        self.data = []
        for obj_code in self.obj_code_list:
            path_i =osp.join(data_path,'{}.npy'.format(obj_code))

            if 'dexrepnet_rollout_trajectory' in self.data_path:
                data = np.load(path_i,allow_pickle=True).tolist()
                self.data.extend(data)
                a=1
            elif 'demonstrations_new7_start_uniform_use_mass_mod_dsam' in self.data_path:
                data = np.load(path_i,allow_pickle=True).item()['grasp_seqs']#[:seq_len,:,6:] #(N,T,28)
                self.data.append(data)

            else:
                data = np.load(path_i,allow_pickle=True).item()
                data = data[self.cfgs.action_type][:,:80,:]
                self.data.append(data)

                a=1

        if 'dexrepnet_rollout_trajectory' in self.data_path:
            self.data = split_and_stack_traj_list(self.data,min_len=50)
            a=1
        # elif 'demonstrations_new7_start_uniform_use_mass_mod_dsam' in self.data_path:
        #     self.data = np.concatenate(self.data,axis=0) #(N*n_obj,T,22)
        else:
            self.data = np.concatenate(self.data,axis=0) #(N*n_obj,T,22)

        self.N,self.T,self.pose_dim=self.data.shape
        a=1

    def poses_to_joints(self, poses):
        bs,D = poses.shape[0],poses.shape[-1]
        if D==22:
            poses =torch.cat([self.shadow_trans.repeat(bs,1), self.shadow_rot.repeat(bs,1), poses],dim=1)
        elif D==28:
            glob_oth6d = robust_compute_ortho6d_from_euler(poses[:,3:6]) #(B, 6)
            poses = torch.cat([poses[:,:3],glob_oth6d, poses[6:]],dim=-1) #(B,31)

        self.hand_model.set_parameters(poses)
        shadow_joints = self.hand_model.get_penetraion_keypoints() #(N*T, 21, 3)
        return shadow_joints

    def poses_to_meshes(self, poses):
        if isinstance(poses,np.ndarray):
            poses = torch.tensor(poses).to(self.device)

        bs,D = poses.shape[0],poses.shape[-1]
        if D==22:
            poses = poses.reshape(-1,22)

            poses =torch.cat([self.shadow_trans.repeat(bs,1), self.shadow_rot.repeat(bs,1), poses],dim=1)
        elif D==28:
            poses = poses.reshape(-1,28)
            glob_oth6d = robust_compute_ortho6d_from_euler(poses[:,3:6]) #(B, 6)
            poses = torch.cat([poses[:,:3],glob_oth6d, poses[:,6:]],dim=-1) #(B,31)


        self.hand_model.set_parameters(poses)

        hand_seq_mesh_list = []
        list_i = []
        N,T  =self.N,self.T
        for i in range(N*T):
            hand_mesh = self.hand_model.get_trimesh_data(i,color='pink')
            hand_mesh = trimesh.util.concatenate(hand_mesh)
            if i%T==0:
                list_i=[]
                list_i.append(hand_mesh)
            else:
                list_i.append(hand_mesh)

            if i%T==T-1:
                hand_seq_mesh_list.append(list_i)

            if len(hand_seq_mesh_list)>=2:
                break

        return hand_seq_mesh_list


    def joint_error(self, joint_gt, joint_rec):
        per_joint_error = torch.norm(joint_gt - joint_rec, dim=-1)  # (N, 21)

        mean_error = per_joint_error.mean(dim=1)*100

        mean_error = mean_error.reshape(self.N,self.T).mean(dim=-1)

        return mean_error

    def SE1(self, traj: torch.Tensor):
        vel = (traj[:, 1:] - traj[:, :-1]) / self.dt  # (B, T-1, J, 3)
        acc = (vel[:, 1:] - vel[:, :-1]) / self.dt  # (B, T-2, J, 3)
        jerk = (acc[:, 1:] - acc[:, :-1]) / self.dt  # (B, T-3, J, 3)

        acc_norm = torch.norm(acc, dim=-1)  # (B, T-2, J)
        jerk_norm = torch.norm(jerk, dim=-1)  # (B, T-3, J)

        Ap = acc_norm.mean(dim=(1, 2))  # (B,)
        Jp = jerk_norm.mean(dim=(1, 2))  # (B,)

        return Ap, Jp

    def SE2(self, traj: torch.Tensor):
        joint_vec = traj[:, 1:] - traj[:, :-1]  # (B, T-1, J, 3)
        joint_vec_norm = joint_vec / (joint_vec.norm(dim=-1, keepdim=True) + 1e-8)  # 避免除零

        joint_cos = (joint_vec_norm[:, 1:] * joint_vec_norm[:, :-1]).sum(dim=-1)  # (B, T-2, J)
        cosp = joint_cos.mean(dim=(1, 2))  # (B,)

        return cosp

    def smooth_error(self, traj):
        vel = (traj[:, 1:] - traj[:, :-1]) / self.dt

        acc = (vel[:, 1:] - vel[:, :-1]) / self.dt

        acc_norm = torch.norm(acc, dim=-1)

        smoothness = acc_norm.mean(dim=(1, 2))

        return smoothness

    def evaluate(self):
        self.vposer_model.eval()

        hand_params = torch.tensor(self.data).reshape(-1,28).to('cuda') #(N*T,28)
        qpos_input = hand_params[:,6:] #(N*T,22)
        qpos_out = self.vposer_model(qpos_input)['pose_shadow22']

        if self.use_gpos:
            glob_oth6d = robust_compute_ortho6d_from_euler(hand_params[:,3:6]) #(B, 6)
            qpos_input =torch.cat([hand_params[:,:3],glob_oth6d, qpos_input],dim=-1) #(B,31)
            qpos_out =torch.cat([hand_params[:,:3],glob_oth6d, qpos_out],dim=-1) #(B,31)

            joints_gt = self.poses_to_joints(qpos_input) #(N*T,21,3)
            joints_out = self.poses_to_joints(qpos_out) #(N*T,21,3)

        else:
            joints_gt = self.poses_to_joints(qpos_input) #(N*T,21,3)
            joints_out = self.poses_to_joints(qpos_out) #(N*T,21,3)

        per_traj_mean_joint_error = self.joint_error(joints_gt,joints_out) #(N)
        per_traj_smooth_error = self.smooth_error(joints_out.reshape(self.N,self.T,-1,3)) *100#(N)
        per_traj_smooth_error_gt = self.smooth_error(joints_gt.reshape(self.N,self.T,-1,3))*100 #(N)

        per_traj_se1_ap, per_traj_se1_jp = self.SE1(joints_out.reshape(self.N, self.T, -1, 3))  # (N)
        per_traj_se1_ap_gt, per_traj_se1_jp_gt = self.SE1(joints_gt.reshape(self.N, self.T, -1, 3))  # (N)
        per_traj_se2 = self.SE2(joints_out.reshape(self.N, self.T, -1, 3))   # (N)
        per_traj_se2_gt = self.SE2(joints_gt.reshape(self.N, self.T, -1, 3))   # (N)

        mean_joint_error = per_traj_mean_joint_error.mean()
        mean_smooth_error = per_traj_smooth_error.mean()
        mean_smooth_error_gt = per_traj_smooth_error_gt.mean()

        print('mean joint error":{:.4f}(cm)'.format(mean_joint_error))
        print('mean smooth error":{:.4f}(cm/s²)'.format(mean_smooth_error))
        print('mean smooth error GT":{:.4f}(cm/s²)'.format(mean_smooth_error_gt))


        print('mean SE1 AP:{:.4f}(m/s²)'.format(per_traj_se1_ap.mean()))
        print('mean SE1 JP:{:.4f}(m/s³)'.format(per_traj_se1_jp.mean()))

        print('mean SE1 AP GT":{:.4f}(m/s²)'.format(per_traj_se1_ap_gt.mean()))
        print('mean SE1 JP GT:{:.4f}(m/s³)'.format(per_traj_se1_jp_gt.mean()))

        print('mean SE2":{:.4f}'.format(per_traj_se2.mean()))
        print('mean SE2 GT":{:.4f}'.format(per_traj_se2_gt.mean()))


        a=1



if __name__ == '__main__':
    from omegaconf import DictConfig
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cfgs =DictConfig({'dataset':'DexGraspNet',

                      'data_path': './dataset/valid',
                      'action_type': 'grasp_seqs', #sim_actions
                      'vposer_ckpt_path': '../ActionDiffusion/bc/saved_models/vposer/epo_6_dex_shadow22_v2v0.149.pt',
                      'dt':0.1,
                      'html_save': False,
                      'seq_len':10,
                      'use_gpos':True,
                      'obj_code_list':['core-bottle-1071fa4cddb2da2fc8724d5673a063a6',
                                       'core-bottle-109d55a137c042f5760315ac3bf2c13e',
                                       'core-bottle-10dff3c43200a7a7119862dbccbaa609',

                        ]

                      })

    reconstor = ShadowVposerReconstructor(cfgs)
    reconstor.evaluate()


