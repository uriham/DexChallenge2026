"""
DemoGrasp 방식 시연 워핑 + reaching 플래너 (O6/GraspM3 판).

reference/grasp.py 의 generate_reaching_plan_idx(:1233-1309) / compute_reference_actions(:1312-1391)
로직을 O6 액션 포맷으로 옮긴 것. IK 분기는 O6엔 불필요해서 제외.

--------------------------------------------------------------------------
검증으로 확정한 O6 손목 기구학 (verify_demo_convention.py, 8/21):

  palm_quat_world = Rx(roll) @ Ry(pitch) @ Rz(yaw)      (scipy 'XYZ' intrinsic, 오차 0)
  palm_pos_world  = (x, y, z) + c,  c = (0, 0, 0.6) 상수 (std ~1e-8)

  즉 액션 앞 6개 (x,y,z,roll,pitch,yaw)는 곧 팜의 world pose이며,
  위치는 상수 오프셋 c만큼, 회전은 정확히 일치한다. FK 계산이 전혀 필요 없다.

  주의: isaacgym의 quat_from_euler_xyz(= extrinsic xyz)는 여기서 **틀린** 컨벤션이다
  (검증에서 offset std 0.0509로 불일치). DemoGrasp 원본 코드를 그대로 쓰면 조용히 어긋난다.
--------------------------------------------------------------------------

워프 계산은 에피소드당 1회만 돌고 (num_envs x T) 규모라 CPU/scipy로 해도 충분히 싸다.
정확성이 검증된 scipy Rotation을 쓰고, 결과 플랜만 GPU 텐서로 올린다.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

# 액션 [Δxyz(3), Δrpy(3), Δfinger(6)]의 tanh 출력 [-1,1]에 곱할 스케일 (논문 값)
WARP_SCALE_TRANS = 0.05   # m
WARP_SCALE_ROT = 1.57     # rad
WARP_SCALE_FINGER = 1.0   # rad

EULER_SEQ = "XYZ"  # intrinsic XYZ = Rx(r)Ry(p)Rz(y), 위 검증으로 확정


def rpy_to_quat(rpy):
    """(...,3) roll/pitch/yaw -> (...,4) xyzw"""
    return R.from_euler(EULER_SEQ, np.asarray(rpy, dtype=np.float64).reshape(-1, 3)).as_quat()


def quat_to_rpy(quat):
    """(...,4) xyzw -> (...,3) roll/pitch/yaw"""
    return R.from_quat(np.asarray(quat, dtype=np.float64).reshape(-1, 4)).as_euler(EULER_SEQ)


def slerp_batch(q0, q1, t):
    """배치 쿼터니언 slerp.
    q0,q1: (N,4) xyzw, t: (T,N) 또는 (N,) -> (T,N,4) 또는 (N,4)"""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    squeeze = t.ndim == 1
    if squeeze:
        t = t[None, :]

    dot = np.sum(q0 * q1, axis=-1)  # (N,)
    q1 = np.where(dot[:, None] < 0, -q1, q1)  # 최단 경로
    dot = np.abs(dot).clip(-1.0, 1.0)

    theta = np.arccos(dot)  # (N,)
    sin_theta = np.sin(theta)
    near_zero = sin_theta < 1e-6

    t_ = t[..., None]  # (T,N,1)
    theta_ = theta[None, :, None]
    sin_theta_ = sin_theta[None, :, None]

    with np.errstate(invalid="ignore", divide="ignore"):
        w0 = np.sin((1.0 - t_) * theta_) / sin_theta_
        w1 = np.sin(t_ * theta_) / sin_theta_
    # 두 쿼터니언이 거의 같으면 단순 lerp로 대체 (0/0 방지)
    w0 = np.where(near_zero[None, :, None], 1.0 - t_, w0)
    w1 = np.where(near_zero[None, :, None], t_, w1)

    out = w0 * q0[None] + w1 * q1[None]
    out /= np.linalg.norm(out, axis=-1, keepdims=True)
    return out[0] if squeeze else out


def quat_angle(q0, q1):
    """두 쿼터니언 사이 각도(rad). q0,q1: (N,4)"""
    dot = np.abs(np.sum(np.asarray(q0) * np.asarray(q1), axis=-1)).clip(-1.0, 1.0)
    return 2.0 * np.arccos(dot)


class DemoReference:
    """시연 pkl을 '물체 프레임 기준' 표현으로 미리 변환해 들고 있는 객체.

    G3(demo_replay_batch)는 물체 위치만 평행이동하고 **물체 회전을 무시**했는데,
    실제로는 오브젝트마다 스폰 회전이 크게 다르다(verify_demo_convention.py로 확인:
    ||R-I||_F 평균 2.06). 여기서는 위치·회전을 모두 물체 프레임에 정렬한다.
    """

    def __init__(self, demo, hand_root_offset):
        """demo: record_demo.py가 저장한 dict
        hand_root_offset: (3,) palm_pos_world - virtual_xyz 상수 c"""
        self.hand_root_offset = np.asarray(hand_root_offset, dtype=np.float64).reshape(3)

        cmd = np.asarray(demo["action_applied"], dtype=np.float64)  # (T,12)
        # index 0은 reset용 더미(전부 0)라 제외. hand_dof(결과)가 아니라 action(명령)을
        # 기준으로 재생해야 한 스텝 밀리지 않는다 (8/21 self-replay에서 확인한 off-by-one).
        cmd = cmd[1:]
        self.T = cmd.shape[0]
        self.t_lift = int(demo["t_lift"]) - 1  # 더미 제거로 인덱스 한 칸 당김
        self.t_lift = int(np.clip(self.t_lift, 1, self.T - 1))

        cmd_xyz = cmd[:, 0:3]
        cmd_rpy = cmd[:, 3:6]
        self.finger = cmd[:, 6:12]  # (T,6)

        # 명령된 팜 world pose
        palm_pos_w = cmd_xyz + self.hand_root_offset[None, :]
        palm_quat_w = rpy_to_quat(cmd_rpy)

        # 시연 당시 물체 초기 pose 기준으로 정렬
        obj_pos0 = np.asarray(demo["obj_pos_world"][0], dtype=np.float64)
        obj_quat0 = np.asarray(demo["obj_quat_world"][0], dtype=np.float64)
        obj_rot0_inv = R.from_quat(obj_quat0).inv()

        self.pos_of = obj_rot0_inv.apply(palm_pos_w - obj_pos0[None, :])  # (T,3)
        self.quat_of = (obj_rot0_inv * R.from_quat(palm_quat_w)).as_quat()  # (T,4)

    def warp(self, obj_pos, obj_quat, actions=None, finger_limits=None):
        """시연을 새 물체 pose에 맞춰 재배치 + (선택) 정책 액션으로 워프.

        obj_pos:  (N,3) 각 env 물체 world 위치
        obj_quat: (N,4) 각 env 물체 world 회전 (xyzw)
        actions:  (N,12) tanh 출력 [-1,1]. None이면 워프 없음(항등).
        finger_limits: (lower(6,), upper(6,)) 관절 한계. None이면 clamp 안 함.

        returns: (T, N, 12) 트래킹 구간 액션
        """
        obj_pos = np.asarray(obj_pos, dtype=np.float64)
        obj_quat = np.asarray(obj_quat, dtype=np.float64)
        N = obj_pos.shape[0]
        T = self.T

        if actions is None:
            d_xyz = np.zeros((N, 3))
            d_rpy = np.zeros((N, 3))
            d_finger = np.zeros((N, 6))
        else:
            a = np.asarray(actions, dtype=np.float64)
            d_xyz = a[:, 0:3] * WARP_SCALE_TRANS
            d_rpy = a[:, 3:6] * WARP_SCALE_ROT
            d_finger = a[:, 6:12] * WARP_SCALE_FINGER

        # --- 1. 물체 프레임 안에서 워프 (회전 -> 평행이동) ---
        warp_rot = R.from_euler(EULER_SEQ, d_rpy)  # (N,)
        pos_of = np.broadcast_to(self.pos_of[:, None, :], (T, N, 3))
        pos_w = np.einsum("nij,tnj->tni", warp_rot.as_matrix(), pos_of) + d_xyz[None, :, :]

        quat_of = R.from_quat(self.quat_of)  # (T,)
        quat_w = np.empty((T, N, 4))
        for n in range(N):  # scipy Rotation은 이 조합의 브로드캐스팅을 지원하지 않음
            quat_w[:, n, :] = (warp_rot[n] * quat_of).as_quat()

        # --- 2. 리프트 구간은 원본의 상승 변위를 그대로 보존 ---
        tl = self.t_lift
        pos_w[tl:] = (
            self.pos_of[tl:, None, :]
            - self.pos_of[tl - 1][None, None, :]
            + pos_w[tl - 1][None, :, :]
        )

        # --- 3. 새 물체 pose로 world 복원 ---
        obj_rot = R.from_quat(obj_quat)
        obj_mat = obj_rot.as_matrix()  # (N,3,3)
        palm_pos_new = np.einsum("nij,tnj->tni", obj_mat, pos_w) + obj_pos[None, :, :]

        palm_quat_new = np.empty((T, N, 4))
        for n in range(N):
            palm_quat_new[:, n, :] = (obj_rot[n] * R.from_quat(quat_w[:, n, :])).as_quat()

        # --- 4. 액션 포맷으로 변환 ---
        act_xyz = palm_pos_new - self.hand_root_offset[None, None, :]
        act_rpy = quat_to_rpy(palm_quat_new.reshape(-1, 4)).reshape(T, N, 3)

        # --- 5. 손가락: 목표 그립 = 시연 그립 + Δ, pre-grasp 구간은 비율 보간 ---
        q0 = self.finger[0]  # (6,)
        q_grasp = self.finger[tl - 1]  # (6,)
        grasp_new = q_grasp[None, :] + d_finger  # (N,6)
        if finger_limits is not None:
            lo, hi = finger_limits
            grasp_new = np.clip(grasp_new, np.asarray(lo)[None, :], np.asarray(hi)[None, :])

        fraction = (grasp_new - q0[None, :]) / (q_grasp - q0 + 1e-6)[None, :]  # (N,6)
        act_finger = np.empty((T, N, 6))
        pre = self.finger[: tl - 1]  # (tl-1, 6)
        act_finger[: tl - 1] = q0[None, None, :] + (pre[:, None, :] - q0[None, None, :]) * fraction[None, :, :]
        act_finger[tl - 1 :] = grasp_new[None, :, :]
        if finger_limits is not None:
            lo, hi = finger_limits
            act_finger = np.clip(act_finger, np.asarray(lo)[None, None, :], np.asarray(hi)[None, None, :])

        return np.concatenate([act_xyz, act_rpy, act_finger], axis=-1).astype(np.float32)


def build_reach(start_wrist6, first_target, max_trans_step=0.01, max_rot_step=0.1, verbose=True):
    """현재 손목 6D -> 워프된 시연 0프레임까지, 필요한 이동/회전 거리에 비례한 스텝 수로 보간.

    reference/utils.py:batch_linear_interpolate_poses(위치 LERP + 회전 SLERP + 스텝 상한)와
    같은 방식. env마다 필요한 스텝 수가 다르므로, 먼저 도착한 env는 마지막 자세를 유지한다.

    start_wrist6: (N,6) 리셋 직후 o6_hand_dof_pos[:,0:6]
    first_target: (N,12) 트래킹 첫 프레임 액션
    returns: (T_reach, N, 12)
    """
    start_wrist6 = np.asarray(start_wrist6, dtype=np.float64)
    first_target = np.asarray(first_target, dtype=np.float64)
    N = start_wrist6.shape[0]

    p0, p1 = start_wrist6[:, 0:3], first_target[:, 0:3]
    q0 = rpy_to_quat(start_wrist6[:, 3:6])
    q1 = rpy_to_quat(first_target[:, 3:6])

    trans_dist = np.linalg.norm(p1 - p0, axis=-1)
    rot_dist = quat_angle(q0, q1)
    n_reach = np.maximum(
        np.ceil(trans_dist / max_trans_step),
        np.ceil(rot_dist / max_rot_step),
    ).clip(1).astype(np.int64)
    T_reach = int(n_reach.max())

    if verbose:
        print("reach: trans(m) min/mean/max = {:.4f}/{:.4f}/{:.4f} | rot(rad) max={:.3f}".format(
            trans_dist.min(), trans_dist.mean(), trans_dist.max(), rot_dist.max()
        ))
        print("reach steps: min={} max={} mean={:.1f}".format(n_reach.min(), n_reach.max(), n_reach.mean()))

    steps = np.arange(1, T_reach + 1)[:, None]  # (T_reach,1)
    frac = np.clip(steps / n_reach[None, :], 0.0, 1.0)  # (T_reach,N), 도착 후엔 1.0 유지

    reach_xyz = p0[None] + frac[..., None] * (p1 - p0)[None]
    reach_quat = slerp_batch(q0, q1, frac)  # (T_reach,N,4)
    reach_rpy = quat_to_rpy(reach_quat.reshape(-1, 4)).reshape(T_reach, N, 3)
    reach_finger = np.broadcast_to(first_target[None, :, 6:12], (T_reach, N, 6))

    return np.concatenate([reach_xyz, reach_rpy, reach_finger], axis=-1).astype(np.float32)


def build_full_plan(demo_ref, obj_pos, obj_quat, start_wrist6, actions=None,
                    finger_limits=None, settle_steps=10, verbose=True):
    """reach + track + settle 을 이어붙인 에피소드 전체 플랜.

    returns: (n_steps, N, 12). index 0은 리셋 직후라 사용되지 않는 더미.
    """
    track = demo_ref.warp(obj_pos, obj_quat, actions=actions, finger_limits=finger_limits)
    reach = build_reach(start_wrist6, track[0], verbose=verbose)
    plan = [reach[0:1], reach, track]
    if settle_steps > 0:
        plan.append(np.repeat(track[-1:], settle_steps, axis=0))
    return np.concatenate(plan, axis=0)
