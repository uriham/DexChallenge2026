import os
import re
import json
import numpy as np
import torch
import transforms3d
import trimesh as tm
import trimesh.sample
import pytorch_kinematics as pk
import pytorch_kinematics.urdf_parser_py.urdf as URDF_PARSER
from pytorch_kinematics.urdf_parser_py.urdf import (URDF, Box, Cylinder, Mesh, Sphere)

from dexgrasp.utils.rot6d import robust_compute_rotation_matrix_from_ortho6d


class HandModel_xhand:
    """
    通用 URDF 手模型（用于 XHand 等）：
    - q 布局: [tx, ty, tz, rot6d(6), joints(J)]
    - 支持：
        * surface points
        * contact points（若有 contact_xhand.json）
        * penetration keypoints（新加，若有 penetration_xhand.json）
    """
    def __init__(self, robot_name, urdf_filename, mesh_path,
                 batch_size=1,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 mesh_nsp=128,
                 hand_scale=1.0,
                 asset_dir=None,
                 allow_missing_contacts=True):
        self.device = device
        self.asset_dir = asset_dir
        self.robot_name = robot_name
        self.batch_size = batch_size
        self.mesh_nsp = mesh_nsp
        self.scale = hand_scale

        # ---------- 路径解析 ----------
        if asset_dir is not None:
            urdf_filename = os.path.join(asset_dir, urdf_filename)
            mesh_path = os.path.join(asset_dir, mesh_path)
        self.urdf_filename = os.path.abspath(urdf_filename)
        self.mesh_path = os.path.abspath(mesh_path)

        # ---------- 解析 URDF（去掉 xml 声明，避免 lxml 报错） ----------
        xml_text = _read_text_strip_xml_decl(self.urdf_filename)
        self.robot = pk.build_chain_from_urdf(xml_text).to(dtype=torch.float, device=self.device)
        self.robot_full = URDF_PARSER.URDF.from_xml_file(self.urdf_filename)
        self.visual_urdf = URDF.from_xml_string(xml_text)

        # 运行时状态
        self.global_translation = None
        self.global_rotation = None
        self.softmax = torch.nn.Softmax(dim=-1)

        # ---------- 读取 contact & penetration json（如果有） ----------
        # contact_xhand.json
        self.contact_point_basis = {}
        self.contact_normals = {}
        self.surface_points = {}
        self.surface_points_normal = {}
        self.mesh_verts = {}
        self.mesh_faces = {}

        contact_json_paths = []
        if asset_dir is not None:
            contact_json_paths.append(os.path.join(asset_dir, 'data/urdf', f'contact_{robot_name}.json'))
        contact_json_paths.append(os.path.join('data/urdf', f'contact_{robot_name}.json'))
        self.contact_point_dict = {}
        loaded_contact = False
        for cj in contact_json_paths:
            if os.path.exists(cj):
                try:
                    self.contact_point_dict = json.load(open(cj, 'r'))
                    loaded_contact = True
                    break
                except Exception:
                    pass
        if not loaded_contact and not allow_missing_contacts:
            raise FileNotFoundError(f'contact_{robot_name}.json not found in: {contact_json_paths}')

        penetration_json_paths = []
        if asset_dir is not None:
            penetration_json_paths.append(os.path.join(asset_dir, f'penetration_{robot_name}.json'))
        penetration_json_paths.append(os.path.join('data/urdf', f'penetration_{robot_name}.json'))

        self.penetration_point_dict = {}
        loaded_penetration = False
        for pj in penetration_json_paths:
            if os.path.exists(pj):
                try:
                    self.penetration_point_dict = json.load(open(pj, 'r'))
                    loaded_penetration = True
                    break
                except Exception:
                    pass

        if not loaded_penetration:
            self.penetration_point_dict = {}

        # 为每个 link 保存局部坐标的 penetration keypoints
        self.penetration_points_per_link = {}

        # ---------- 构建每个 link 的 surface 点和 mesh ----------
        for i_link, link in enumerate(self.visual_urdf.links):
            if len(link.visuals) == 0:
                continue

            geom = link.visuals[0].geometry
            mesh = None
            if isinstance(geom, Mesh):
                filename = geom.filename
                mesh_file = _resolve_mesh_file(self.mesh_path, filename)
                mesh = tm.load(mesh_file, force='mesh', process=False)
            elif isinstance(geom, Cylinder):
                mesh = tm.primitives.Cylinder(radius=geom.radius, height=geom.length)
            elif isinstance(geom, Box):
                mesh = tm.primitives.Box(extents=geom.size)
            elif isinstance(geom, Sphere):
                mesh = tm.primitives.Sphere(radius=geom.radius)
            else:
                raise NotImplementedError(f'Unsupported geometry type: {type(geom)}')

            # ---- visual transform: scale / rotation / translation ----
            raw_scale = getattr(geom, 'scale', None)
            if raw_scale is None:
                scale = np.array([[1.0, 1.0, 1.0]], dtype=float)
            else:
                try:
                    arr = np.array(raw_scale, dtype=float).reshape(-1)
                    if arr.size == 1:
                        s = float(arr[0])
                        scale = np.array([[s, s, s]], dtype=float)
                    else:
                        scale = np.array([arr[:3]], dtype=float)
                except Exception:
                    scale = np.array([[1.0, 1.0, 1.0]], dtype=float)

            try:
                rotation = transforms3d.euler.euler2mat(*link.visuals[0].origin.rpy)
                translation = np.reshape(link.visuals[0].origin.xyz, [1, 3])
            except AttributeError:
                rotation = transforms3d.euler.euler2mat(0, 0, 0)
                translation = np.array([[0, 0, 0]])

            # ---- surface samples & normals ----
            sample_count = 64 if self.robot_name == 'shadowhand' else 128
            pts, face_idx = trimesh.sample.sample_surface(mesh=mesh, count=sample_count)
            pts_n = np.array([mesh.face_normals[x] for x in face_idx], dtype=float)

            pts = pts * scale
            pts = (rotation @ pts.T).T + translation
            pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=-1)

            pts_n = np.concatenate([pts_n, np.ones((len(pts_n), 1))], axis=-1)

            self.surface_points[link.name] = torch.from_numpy(pts_h).to(self.device).float().unsqueeze(0).repeat(self.batch_size, 1, 1)
            self.surface_points_normal[link.name] = torch.from_numpy(pts_n).to(self.device).float().unsqueeze(0).repeat(self.batch_size, 1, 1)

            # ---- 缓存 mesh 顶点/面，用于可视化 ----
            v = np.array(mesh.vertices) * scale
            v = (rotation @ v.T).T + translation
            self.mesh_verts[link.name] = v
            self.mesh_faces[link.name] = np.array(mesh.faces)

            # ---- contact points（如果有 contact_xhand.json）----
            if link.name in self.contact_point_dict:
                cpb = np.array(self.contact_point_dict[link.name])
                if cpb.ndim > 1:
                    cpb = cpb[np.random.randint(cpb.shape[0], size=1)][0]
                cp_basis = mesh.vertices[cpb] * scale
                cp_basis = (rotation @ cp_basis.T).T + translation
                cp_basis = torch.cat(
                    [torch.from_numpy(cp_basis).to(self.device).float(),
                     torch.ones((len(cp_basis), 1), device=self.device)],
                    dim=-1
                )
                self.contact_point_basis[link.name] = cp_basis.unsqueeze(0).repeat(self.batch_size, 1, 1)
                v1 = cp_basis[1, :3] - cp_basis[0, :3]
                v2 = cp_basis[2, :3] - cp_basis[0, :3]
                v1 = v1 / torch.norm(v1)
                v2 = v2 / torch.norm(v2)
                nrm = torch.cross(v1, v2).view(1, 3)
                self.contact_normals[link.name] = nrm.unsqueeze(0).repeat(self.batch_size, 1, 1)

            # ---- penetration keypoints（新加）----
            if link.name in self.penetration_point_dict:
                pp = np.array(self.penetration_point_dict[link.name], dtype=float)
                if pp.size == 0:
                    kp = torch.zeros((0, 3), dtype=torch.float32, device=self.device)
                else:
                    pp = pp.reshape(-1, 3)
                    kp = torch.from_numpy(pp).float().to(self.device)
                # 这里的 kp 仍然是 link 局部坐标（不额外乘 R/T），
                # 后面 get_penetraion_keypoints 里用 current_status[link].get_matrix() 进行变换。
            else:
                kp = torch.zeros((0, 3), dtype=torch.float32, device=self.device)

            self.penetration_points_per_link[link.name] = kp

        # 统计总的 keypoint 数量（可选）
        self.n_keypoints = int(sum(kp.shape[0] for kp in self.penetration_points_per_link.values()))

        # ---------- 关节上下限 ----------
        self.revolute_joints = []
        for j in self.robot_full.joints:
            if j.joint_type in ('revolute', 'continuous', 'prismatic'):
                self.revolute_joints.append(j)

        joint_names = self.robot.get_joint_parameter_names()
        lower_list, upper_list = [], []
        for name in joint_names:
            j = None
            for jj in self.revolute_joints:
                if jj.name == name:
                    j = jj
                    break
            if j is None or j.limit is None:
                if j is not None and j.joint_type == 'prismatic':
                    lower, upper = -0.05, 0.05
                else:
                    lower, upper = -np.pi, np.pi
            else:
                lower, upper = j.limit.lower, j.limit.upper
                if lower is None or upper is None:
                    lower, upper = (-np.pi, np.pi) if j.joint_type != 'prismatic' else (-0.05, 0.05)
            lower_list.append(lower)
            upper_list.append(upper)

        self.revolute_joints_q_lower = torch.tensor(lower_list, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.batch_size, 1)
        self.revolute_joints_q_upper = torch.tensor(upper_list, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.batch_size, 1)

        self.current_status = None

    # ------------------- Kinematics & geometry queries -------------------
    def update_kinematics(self, q):
        """
        q: (B, 9+J) -> [tx,ty,tz, rot6d(6), joints(J)]
        """
        self.global_translation = q[:, :3]
        self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(q[:, 3:9])
        self.current_status = self.robot.forward_kinematics(q[:, 9:])

    def get_surface_points(self, q=None, downsample=True):
        if q is not None:
            self.update_kinematics(q)
        surface_points = []
        for link_name in self.surface_points:
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(
                torch.matmul(
                    trans_matrix,
                    self.surface_points[link_name].transpose(1, 2)
                ).transpose(1, 2)[..., :3]
            )
        surface_points = torch.cat(surface_points, 1)
        surface_points = torch.matmul(self.global_rotation, surface_points.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        return surface_points * self.scale

    def get_surface_points_new(self, q=None, downsample=True):
        return self.get_surface_points(q=q, downsample=downsample)

    def get_surface_points_and_normals(self, q=None):
        if q is not None:
            self.update_kinematics(q=q)
        surface_points = []
        surface_normals = []
        for link_name in self.surface_points:
            trans_matrix = self.current_status[link_name].get_matrix()
            surface_points.append(
                torch.matmul(
                    trans_matrix,
                    self.surface_points[link_name].transpose(1, 2)
                ).transpose(1, 2)[..., :3]
            )
            surface_normals.append(
                torch.matmul(
                    trans_matrix,
                    self.surface_points_normal[link_name].transpose(1, 2)
                ).transpose(1, 2)[..., :3]
            )
        surface_points = torch.cat(surface_points, 1)
        surface_normals = torch.cat(surface_normals, 1)
        surface_points = torch.matmul(self.global_rotation, surface_points.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        surface_normals = torch.matmul(self.global_rotation, surface_normals.transpose(1, 2)).transpose(1, 2)
        return surface_points * self.scale, surface_normals

    def get_key_points(self, links, q=None):
        if q is not None:
            self.update_kinematics(q)
        key_points = []
        for link_name in links:
            trans_matrix = self.current_status[link_name].get_matrix()
            key_points.append(
                torch.matmul(
                    trans_matrix,
                    self.surface_points[link_name].mean(dim=1, keepdim=True).transpose(1, 2)
                ).transpose(1, 2)[..., :3]
            )
        key_points = torch.cat(key_points, 1)
        key_points = torch.matmul(self.global_rotation, key_points.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        return (key_points * self.scale).squeeze(0)

    def get_meshes_from_q(self, q=None, i=0, color=[0, 0.5, 0.5]):
        data = []
        if q is not None:
            self.update_kinematics(q)
        for idx, link_name in enumerate(self.mesh_verts):
            trans_matrix = self.current_status[link_name].get_matrix()
            trans_matrix = trans_matrix[min(len(trans_matrix) - 1, i)].detach().cpu().numpy()
            v = self.mesh_verts[link_name]
            transformed_v = np.concatenate([v, np.ones((len(v), 1))], axis=-1)
            transformed_v = np.matmul(trans_matrix, transformed_v.T).T[..., :3]
            transformed_v = np.matmul(self.global_rotation[i].detach().cpu().numpy(), transformed_v.T).T + np.expand_dims(self.global_translation[i].detach().cpu().numpy(), 0)
            transformed_v = transformed_v * self.scale
            f = self.mesh_faces[link_name]
            data.append(tm.Trimesh(vertices=transformed_v, faces=f))
        data = tm.util.concatenate(data)
        vertex_color = np.ones((len(data.vertices), 4)) * 254
        vertex_color[:, :3] *= np.asarray(color)
        data.visual.vertex_colors = vertex_color
        return data

    # ---------- 新增：penetration keypoints ----------
    def get_penetraion_keypoints(self, q=None):
        """
        与 Shadow Hand 版本基本一致：
        - 先用 current_status[link].get_matrix() 把各 link 的局部 keypoints 变到手坐标
        - 再叠加 global_rotation / global_translation
        - 返回 (B, N_keypoints, 3)
        """
        if q is not None:
            self.update_kinematics(q)

        if self.current_status is None or self.global_translation is None or self.global_rotation is None:
            raise RuntimeError("Must call update_kinematics(q) or传入 q 之后才能调用 get_penetraion_keypoints")

        batch_size = self.global_translation.shape[0]
        points = []

        for link_name, local_kp in self.penetration_points_per_link.items():
            if local_kp is None or local_kp.numel() == 0:
                continue
            # local_kp: (K, 3) in link frame
            K = local_kp.shape[0]
            # (K,4) -> (B,K,4)
            local_h = torch.cat(
                [local_kp, torch.ones((K, 1), device=self.device, dtype=torch.float32)],
                dim=-1
            )  # (K,4)
            local_h = local_h.unsqueeze(0).expand(batch_size, K, 4)  # (B,K,4)

            trans_matrix = self.current_status[link_name].get_matrix()  # (B,4,4)
            # (B,4,4) x (B,4,K) -> (B,4,K) -> (B,K,3)
            kp = torch.matmul(trans_matrix, local_h.transpose(2, 1)).transpose(2, 1)[..., :3]
            points.append(kp)

        if len(points) == 0:
            # 没有 keypoints 就返回一个 empty tensor
            return torch.zeros((batch_size, 0, 3), dtype=torch.float32, device=self.device)

        points = torch.cat(points, dim=1)  # (B, N_total, 3)
        points = torch.matmul(self.global_rotation, points.transpose(2, 1)).transpose(2, 1) + self.global_translation.unsqueeze(1)
        return points * self.scale

    # ---------- Contact queries (保留原有) ----------
    def get_contact_points(self, contact_point_part_indices, contact_point_weights, q=None):
        if len(self.contact_point_basis) == 0:
            raise RuntimeError('No contact points available: contact json not provided.')
        contact_point_weights = self.softmax(contact_point_weights)
        if q is not None:
            self.update_kinematics(q)
        contact_point_basis_transformed = []
        for link_name in self.contact_point_basis:
            trans_matrix = self.current_status[link_name].get_matrix()
            cp_basis = self.contact_point_basis[link_name]
            cp_basis_transformed = torch.matmul(trans_matrix, cp_basis.transpose(1, 2)).transpose(1, 2)[..., :3]
            contact_point_basis_transformed.append(cp_basis_transformed)
        contact_point_basis_transformed = torch.stack(contact_point_basis_transformed, 1)  # B x J x 4 x 3
        contact_point_basis_transformed = contact_point_basis_transformed[
            torch.arange(0, len(contact_point_basis_transformed), device=self.device).unsqueeze(1).long(),
            contact_point_part_indices.long()
        ]
        contact_point_basis_transformed = (contact_point_basis_transformed * contact_point_weights.unsqueeze(-1)).sum(2)
        contact_point_basis_transformed = torch.matmul(self.global_rotation, contact_point_basis_transformed.transpose(1, 2)).transpose(1, 2) + self.global_translation.unsqueeze(1)
        return contact_point_basis_transformed * self.scale


def _read_text_strip_xml_decl(path):
    with open(path, 'r', encoding='utf-8') as f:
        xml_text = f.read()
    # strip BOM
    if xml_text.startswith(u"\ufeff"):
        xml_text = xml_text.lstrip(u"\ufeff")
    # remove XML declaration (<?xml ...?>)
    xml_text = re.sub(r'^\s*<\?xml[^>]*\?>', '', xml_text)
    return xml_text


def _resolve_mesh_file(mesh_root, filename):
    # If filename is absolute, use it directly
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    fname = filename
    if fname.startswith('package://'):
        fname = fname.split('package://', 1)[1]
        parts = fname.split('/', 1)
        if len(parts) == 2:
            fname = parts[1]
    candidate = os.path.join(mesh_root, fname)
    if os.path.exists(candidate):
        return candidate
    return os.path.abspath(candidate)


if __name__ == '__main__':
    """
    简单测试：打印 link 名、关节名，并导出一个带 surface points 的 HTML（如果安装了 plotly）
    """
    try:
        from plotly import graph_objects as go
        has_plotly = True
    except Exception:
        has_plotly = False

    robot_name = 'xhand'
    asset_dir = '/home/zbh/WorkSpace/factory_dexmanip/assets/xhand_right/urdf'
    urdf_rel = 'xhand_right.urdf'
    mesh_rel = ''   # 如果有单独 meshes 目录就改这里

    device = 'cpu'

    model = HandModel_xhand(
        robot_name=robot_name,
        urdf_filename=urdf_rel,
        mesh_path=mesh_rel,
        batch_size=1,
        device=torch.device(device),
        mesh_nsp=128,
        hand_scale=1.0,
        asset_dir=asset_dir,
        allow_missing_contacts=True,
    )

    print('=== visual_urdf link names ===')
    for link in model.visual_urdf.links:
        print('  ', link.name)

    print('=== robot_full link names ===')
    for link in model.robot_full.links:
        print('  ', link.name)

    joint_names = model.robot.get_joint_parameter_names()
    print('=== joint parameter names ===')
    print(joint_names)
    print('[ok] DOF =', len(joint_names))

    lower = model.revolute_joints_q_lower[0].cpu().numpy()
    upper = model.revolute_joints_q_upper[0].cpu().numpy()
    mid = 0.5 * (lower + upper)

    J = len(joint_names)
    q = torch.zeros((1, 9 + J), dtype=torch.float32, device=device)
    q[:, :3] = torch.tensor([0.0, 0.0, 0.0])
    q[:, 3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])  # rot6d = I
    q[:, 9:] = torch.from_numpy(mid)

    surf_pts = model.get_surface_points(q=q).cpu().squeeze(0).numpy()
    print('[ok] surface points:', surf_pts.shape)

    # 测试 penetration keypoints（若有 json）
    try:
        kps = model.get_penetraion_keypoints(q=q).cpu().squeeze(0).numpy()
        print('[ok] penetration keypoints:', kps.shape)
    except Exception as e:
        print('[warn] get_penetraion_keypoints failed:', e)

    if has_plotly:
        mesh_trimesh = model.get_meshes_from_q(q=q, color=[0.2, 0.6, 1.0])
        fig = go.Figure([
            go.Mesh3d(
                x=mesh_trimesh.vertices[:, 0],
                y=mesh_trimesh.vertices[:, 1],
                z=mesh_trimesh.vertices[:, 2],
                i=mesh_trimesh.faces[:, 0],
                j=mesh_trimesh.faces[:, 1],
                k=mesh_trimesh.faces[:, 2],
                color='lightblue',
                opacity=0.85
            ),
            go.Scatter3d(
                x=surf_pts[:, 0],
                y=surf_pts[:, 1],
                z=surf_pts[:, 2],
                mode='markers',
                marker=dict(size=2),
                name='surface_pts'
            )
        ])
        out_html = './test_xhand.html'
        fig.write_html(out_html)
        print(f'[ok] wrote: {out_html}')
    else:
        print('[note] Plotly not installed; skipped HTML export.')

    # import os
    # import torch
    #
    # # ===== 手工指定 Wuji hand 的 URDF 路径 =====
    # wuji_asset_dir = "/assets/wujihand_urdf/urdf"
    # wuji_urdf_name = "right.urdf"
    #
    # if not os.path.exists(os.path.join(wuji_asset_dir, wuji_urdf_name)):
    #     raise FileNotFoundError(
    #         f"Wuji URDF not found: {os.path.join(wuji_asset_dir, wuji_urdf_name)}"
    #     )
    #
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"[WujiTest] Using device: {device}")
    #
    # wuji_model = HandModel_xhand(
    #     robot_name="wuji",
    #     urdf_filename=wuji_urdf_name,
    #     mesh_path="../",
    #     batch_size=1,
    #     device=device,
    #     mesh_nsp=128,                # 每个 link 采样点数，可以根据需要改
    #     hand_scale=1.0,
    #     asset_dir=wuji_asset_dir,
    #     allow_missing_contacts=True,
    # )
    #
    #
    # print("\n[WujiTest] ===== Wuji hand loaded =====")
    # joint_names = wuji_model.robot.get_joint_parameter_names()
    # print(f"[WujiTest] Number of DOFs (joints): {len(joint_names)}")
    # print("[WujiTest] Joint names:")
    # for i, name in enumerate(joint_names):
    #     print(f"  [{i:02d}] {name}")
    #
    # print('=== visual_urdf link names ===')
    # for link in wuji_model.visual_urdf.links:
    #     print('  ', link.name)
    #
    # print('=== robot_full link names ===')
    # for link in wuji_model.robot_full.links:
    #     print('  ', link.name)
    #
    # # ===== 构造一个全 0 的 q，并测试关键点函数 =====
    # # HandModel_xhand 里 q 的布局通常为：
    # #   [ tx, ty, tz, rot6d(6), joints(J) ]
    # J = len(joint_names)
    # q_zero = torch.zeros((1, 9 + J), dtype=torch.float32, device=device)
    #
    # print("\n[WujiTest] Forward q_zero and compute penetration keypoints...")
    # try:
    #     kps = wuji_model.get_penetraion_keypoints(q_zero)  # (B, K, 3) 或 (K,3)
    #     if isinstance(kps, torch.Tensor):
    #         kps = kps[0] if kps.ndim == 3 else kps  # 取 batch 中第一只手
    #     print(f"[WujiTest] keypoints shape: {kps.shape}")
    #     if kps.shape[0] > 0:
    #         print(f"[WujiTest] first 5 keypoints:\n{kps[:5].detach().cpu().numpy()}")
    #     else:
    #         print("[WujiTest] No keypoints returned (可能是没加载 contact/penetration json)")
    # except Exception as e:
    #     print("[WujiTest] Error when calling get_penetraion_keypoints:", e)
    #
    # print("\n[WujiTest] Done.")
