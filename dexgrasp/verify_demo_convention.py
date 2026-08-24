"""
M5-0 선행 검증 (시뮬레이터 불필요, scipy만 사용).

1) 오일러 컨벤션 확정:
   기록된 시연에는 hand_dof[:,3:6](가상관절 roll/pitch/yaw)과 palm_quat_world(실제 쿼터니언)가
   둘 다 들어있다. 여러 후보 컨벤션으로 rpy->quat을 만들어보고,
   "q_cand(t)^-1 * palm_quat(t) = 상수 offset"이 성립하는 컨벤션을 찾는다.
   상수가 나오는 컨벤션이 곧 IsaacGym이 가상관절 체인을 해석하는 방식이고,
   워프에서 회전을 합성할 때 그 컨벤션을 써야 한다.

2) obj_rotmat 분산 확인:
   G3 실패의 미해결 의심 사항 - 오브젝트마다 스폰 orientation이 다른가?
   다르다면 "물체 회전 랜덤화가 없으니 시연 rpy를 그대로 써도 된다"는 G3의 가정이 틀린 것.

사용법 (dexgrasp/ 에서):
    python verify_demo_convention.py
"""
import glob
import os.path as osp
import pickle

import numpy as np
from scipy.spatial.transform import Rotation as R

THIS_DEXGRASP_DIR = osp.realpath(osp.dirname(__file__))
DEMO_PKL = osp.join(THIS_DEXGRASP_DIR, "demos", "core-bottle-be16ada66829940a451786f3cbfd6769_traj0.pkl")
SOURCE_PREPROC_DIR = "/data/DexGraspMotionChallenge2026/dataset_shinwoo_preproc/train"


def quat_conj(q):
    """q: (...,4) xyzw"""
    out = q.copy()
    out[..., :3] *= -1.0
    return out


def quat_mul(a, b):
    """a,b: (...,4) xyzw. returns a*b"""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def canonicalize_sign(q):
    """쿼터니언 q와 -q는 같은 회전이므로 w>=0으로 부호 통일."""
    q = q.copy()
    flip = q[..., 3] < 0
    q[flip] *= -1.0
    return q


# 후보 컨벤션들. rpy=(roll,pitch,yaw) -> quat(xyzw)
# 대문자=intrinsic, 소문자=extrinsic (scipy 규약)
CANDIDATES = {
    # isaacgym quat_from_euler_xyz(r,p,y)와 동일: Rz(y)Ry(p)Rx(r)
    "isaacgym quat_from_euler_xyz  = extrinsic xyz [r,p,y]": lambda rpy: R.from_euler("xyz", rpy).as_quat(),
    # URDF 직렬체인 roll->pitch->yaw 예측: Rx(r)Ry(p)Rz(y)
    "URDF chain 예측               = extrinsic zyx [y,p,r]": lambda rpy: R.from_euler("zyx", rpy[:, ::-1]).as_quat(),
    "intrinsic XYZ [r,p,y]": lambda rpy: R.from_euler("XYZ", rpy).as_quat(),
    "intrinsic ZYX [y,p,r]": lambda rpy: R.from_euler("ZYX", rpy[:, ::-1]).as_quat(),
}


def check_convention(rpy, palm_quat):
    print("=" * 78)
    print("1) 오일러 컨벤션 검증")
    print("=" * 78)
    print("프레임 수: {}".format(rpy.shape[0]))
    print()

    results = []
    for name, fn in CANDIDATES.items():
        q_cand = canonicalize_sign(fn(rpy).astype(np.float64))
        # offset(t) = q_cand(t)^-1 * palm_quat(t)  (상수면 정답 컨벤션)
        offset = canonicalize_sign(quat_mul(quat_conj(q_cand), palm_quat))
        spread = offset.std(axis=0).max()  # 성분별 표준편차 중 최대
        results.append((spread, name, offset.mean(axis=0)))
        print("{:55s} offset std(max)={:.6f}".format(name, spread))

    results.sort()
    best_spread, best_name, best_offset = results[0]
    print()
    print("최적 후보: {}".format(best_name))
    print("  offset std(max) = {:.6f}".format(best_spread))
    print("  offset (평균, xyzw) = {}".format(np.round(best_offset, 5)))
    print()
    if best_spread < 1e-3:
        print("  => PASS. 이 컨벤션으로 회전 워프를 합성하면 된다.")
    elif best_spread < 1e-2:
        print("  => 애매함. 수치오차 수준을 넘음 - 시연 프레임 수를 늘리거나 다른 시연으로 재확인 권장.")
    else:
        print("  => FAIL. 어떤 후보도 상수 offset을 못 만듦.")
        print("     회전 워프를 빼고 9D 액션(Δxyz+Δfinger)으로 진행할 것.")
    return best_spread, best_name


def check_obj_rotmat_variance(n_sample=40):
    print()
    print("=" * 78)
    print("2) obj_rotmat 분산 확인 (G3 실패 원인 후보)")
    print("=" * 78)
    files = sorted(glob.glob(osp.join(SOURCE_PREPROC_DIR, "*.npy")))[:n_sample]
    if not files:
        print("전처리 파일을 못 찾음: {}".format(SOURCE_PREPROC_DIR))
        return

    rotmats = []
    for f in files:
        d = np.load(f, allow_pickle=True).item()
        rotmats.append(np.asarray(d["obj_rotmat"][0], dtype=np.float64))
    rotmats = np.stack(rotmats)  # (n,3,3)

    identity_dist = np.linalg.norm(rotmats - np.eye(3)[None], axis=(1, 2))
    pairwise_spread = rotmats.std(axis=0).max()

    print("샘플 오브젝트 수: {}".format(len(files)))
    print("||R - I||_F : min={:.4f} mean={:.4f} max={:.4f}".format(
        identity_dist.min(), identity_dist.mean(), identity_dist.max()
    ))
    print("오브젝트 간 R 성분 표준편차(max): {:.4f}".format(pairwise_spread))
    print()
    if pairwise_spread < 1e-6:
        print("  => 모든 오브젝트가 동일 orientation. G3의 'rpy 그대로' 가정은 유효했음.")
        print("     (따라서 G3 14%의 원인은 다른 데 있음)")
    else:
        print("  => 오브젝트마다 스폰 회전이 다름. G3는 시연의 손목 rpy를 그대로 썼으므로,")
        print("     물체 회전 차이만큼 손이 어긋난 채로 접근했을 가능성이 큼.")
        print("     => M5의 회전 워프가 이 차이를 흡수해줄 여지가 있다는 뜻(M5에 유리한 신호).")


def main():
    with open(DEMO_PKL, "rb") as f:
        demo = pickle.load(f)
    print("demo: object={} success={} t_lift={} T={}".format(
        demo["object_code"], demo["success"], demo["t_lift"], demo["T"]
    ))
    print()

    rpy = np.asarray(demo["hand_dof"][:, 3:6], dtype=np.float64)
    palm_quat = canonicalize_sign(np.asarray(demo["palm_quat_world"], dtype=np.float64))

    check_convention(rpy, palm_quat)
    check_obj_rotmat_variance()


if __name__ == "__main__":
    main()
