import torch
from dexgrasp.utils.HandModel_xhand import HandModel_xhand


class HandModel_Linkerhand(HandModel_xhand):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 原始 O6 手指共有 11 个 revolute 关节。
        # 其中 6 个主动受控关节的准确索引为：
        # 0: thumb_yaw, 1: thumb_pitch, 3: index, 5: middle, 7: ring, 9: pinky
        self.active_joint_indices = [0, 1, 3, 5, 7, 9]

        # 覆盖基类的自由度边界，只提取这 6 个主动自由度用于截断和标准化
        self.revolute_joints_q_lower_full = self.revolute_joints_q_lower.clone()
        self.revolute_joints_q_upper_full = self.revolute_joints_q_upper.clone()

        self.revolute_joints_q_lower = self.revolute_joints_q_lower_full[:, self.active_joint_indices]
        self.revolute_joints_q_upper = self.revolute_joints_q_upper_full[:, self.active_joint_indices]

    def update_kinematics(self, q):
        """
        q 的输入布局为 [tx, ty, tz, rot6d(6), joints(6)] (共 15 维)
        底层 robot.forward_kinematics 需要 11 个 joints。
        """
        q_global = q[:, :9]  # 根节点平移 + 旋转
        q_active = q[:, 9:]  # 6 个主动关节角

        # 构建完整的 11 维关节角
        q_full = torch.zeros((q.shape[0], 11), device=self.device, dtype=q.dtype)

        # 拇指
        q_full[:, 0] = q_active[:, 0]
        q_full[:, 1] = q_active[:, 1]
        q_full[:, 2] = q_active[:, 1] * 1.86  # thumb_ip (mimic)

        # 食指
        q_full[:, 3] = q_active[:, 2]
        q_full[:, 4] = q_active[:, 2] * 0.89  # index_dip

        # 中指
        q_full[:, 5] = q_active[:, 3]
        q_full[:, 6] = q_active[:, 3] * 0.89  # middle_dip

        # 无名指
        q_full[:, 7] = q_active[:, 4]
        q_full[:, 8] = q_active[:, 4] * 0.89  # ring_dip

        # 小指
        q_full[:, 9] = q_active[:, 5]
        q_full[:, 10] = q_active[:, 5] * 0.89  # pinky_dip

        # 重新拼合: q_global (9维) + q_full (11维) = 20维，传给父类
        q_new = torch.cat([q_global, q_full], dim=-1)

        return super().update_kinematics(q_new)