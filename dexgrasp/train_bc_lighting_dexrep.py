import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import os.path as osp
import sys

THIS_DEXGRASP_DIR = osp.realpath(osp.dirname(__file__))
THIS_REPO_DIR = osp.realpath(osp.join(THIS_DEXGRASP_DIR, ".."))
for _path in [THIS_REPO_DIR, THIS_DEXGRASP_DIR]:
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

import csv
import yaml

from isaacgym.torch_utils import *
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader, WeightedRandomSampler

from ActionDiffusion.bc.dataset.graspm3_dexrep import GraspM3DexRepDataset
from ActionDiffusion.bc.model.policy.lhm_policy import LitBCModel


def _env_flag(name):
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}

def _runtime_cfg(args):
    return args.get("train_runtime", {}) or {}

def _runtime_get(args, key, default=None):
    return _runtime_cfg(args).get(key, default)

def _runtime_flag(args, key, env_name=None, default=False):
    if env_name and os.environ.get(env_name) is not None:
        return _env_flag(env_name)
    value = _runtime_get(args, key, default)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)

def _runtime_int(args, key, env_name=None, default=None):
    if env_name and os.environ.get(env_name):
        return int(os.environ[env_name])
    value = _runtime_get(args, key, default)
    return None if value is None else int(value)

def _runtime_float(args, key, env_name=None, default=None):
    if env_name and os.environ.get(env_name):
        return float(os.environ[env_name])
    value = _runtime_get(args, key, default)
    return None if value is None else float(value)

def _runtime_str(args, key, env_name=None, default=None):
    if env_name and os.environ.get(env_name):
        return os.environ[env_name]
    value = _runtime_get(args, key, default)
    if value in (None, ""):
        return default
    return str(value)

def _runtime_list(args, key, env_name=None):
    if env_name and os.environ.get(env_name):
        return [
            value.strip()
            for value in os.environ[env_name].split(",")
            if value.strip()
        ]
    value = _runtime_get(args, key, None)
    if value in (None, ""):
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def configure_o6_dexrep_obs_dims(args, env_args, dataset=None):
    if getattr(args, "o6_obs_mode", None) != "dexrep":
        return

    args.obs_type = "dexrep"
    env_args.env.o6_policy_obs_mode = "dexrep"
    env_args.env.obs_dim.prop = int(args.obs_dim)
    if getattr(args, "dexrep_sensor_dim", None) is not None:
        env_args.env.obs_dim.dexrep_sensor = int(args.dexrep_sensor_dim)
    if getattr(args, "dexrep_pnl_dim", None) is not None:
        env_args.env.obs_dim.dexrep_pnl = int(args.dexrep_pnl_dim)

    if dataset is None:
        return

    observed_dim = int(dataset.data["obs"].shape[-1])
    expected_dim = int(env_args.env.obs_dim.prop + env_args.env.obs_dim.dexrep_sensor + env_args.env.obs_dim.dexrep_pnl)
    if observed_dim == expected_dim:
        return

    dexrep_dim = observed_dim - int(env_args.env.obs_dim.prop)
    if dexrep_dim <= 0:
        raise ValueError(
            "o6 dexrep obs dim {} is not larger than prop dim {}; "
            "check preprocessing and o6_obs_mode".format(observed_dim, int(env_args.env.obs_dim.prop))
        )
    if getattr(args, "o6_obs_mode", None) == "dexrep":
        # O6 uses 10 DexRep query sites: 10 * (normal xyz + distance) + 1000
        # ambient occupancy = 1040 sensor dims, and 10 * 64 = 640 local point
        # feature dims. Preserve this semantic split when the total dim matches.
        o6_sensor_dim = 1040
        o6_pnl_dim = dexrep_dim - o6_sensor_dim
        if o6_pnl_dim > 0:
            env_args.env.obs_dim.dexrep_sensor = o6_sensor_dim
            env_args.env.obs_dim.dexrep_pnl = o6_pnl_dim
            if hasattr(args, "dexrep_sensor_dim"):
                args.dexrep_sensor_dim = o6_sensor_dim
            if hasattr(args, "dexrep_pnl_dim"):
                args.dexrep_pnl_dim = o6_pnl_dim
            print(
                "auto_o6_dexrep_obs_dims: observed_dim={}, expected_dim={}, "
                "prop={}, dexrep_sensor={}, dexrep_pnl={}, bn_type={}".format(
                    observed_dim,
                    expected_dim,
                    int(env_args.env.obs_dim.prop),
                    int(env_args.env.obs_dim.dexrep_sensor),
                    int(env_args.env.obs_dim.dexrep_pnl),
                    args.encoder.bn_type,
                )
            )
            return
    old_sensor = int(env_args.env.obs_dim.dexrep_sensor)
    old_pnl = int(env_args.env.obs_dim.dexrep_pnl)
    if old_pnl > 0 and dexrep_dim >= old_pnl:
        env_args.env.obs_dim.dexrep_pnl = old_pnl
        env_args.env.obs_dim.dexrep_sensor = dexrep_dim - old_pnl
    else:
        env_args.env.obs_dim.dexrep_sensor = dexrep_dim
        env_args.env.obs_dim.dexrep_pnl = 0
        args.encoder.bn_type = "null"
    print(
        "auto_o6_dexrep_obs_dims: observed_dim={}, expected_dim={}, "
        "prop={}, dexrep_sensor={}, dexrep_pnl={}, bn_type={}".format(
            observed_dim,
            expected_dim,
            int(env_args.env.obs_dim.prop),
            int(env_args.env.obs_dim.dexrep_sensor),
            int(env_args.env.obs_dim.dexrep_pnl),
            args.encoder.bn_type,
        )
    )

def assert_clean_runtime_paths():
    from pathlib import Path

    if Path.cwd().resolve() != Path(__file__).resolve().parent:
        raise RuntimeError("train_bc_lighting_dexrep.py must run from this branch's dexgrasp directory")

def build_object_balanced_sampler(dataset):
    if not dataset.is_flat:
        raise ValueError("object-balanced sampler expects batch_seq_flat=True")

    seq_obj_ids = np.asarray(dataset.data["obj_code_idx"], dtype=np.int64)
    frame_obj_ids = np.repeat(seq_obj_ids, int(dataset.num_frame))
    if len(frame_obj_ids) != len(dataset):
        raise ValueError(
            "frame/object mapping length mismatch: {} vs dataset {}".format(
                len(frame_obj_ids),
                len(dataset),
            )
        )

    obj_ids, obj_counts = np.unique(frame_obj_ids, return_counts=True)
    obj_count_map = {int(obj_id): int(count) for obj_id, count in zip(obj_ids, obj_counts)}
    sample_weights = np.asarray(
        [1.0 / obj_count_map[int(obj_id)] for obj_id in frame_obj_ids],
        dtype=np.float64,
    )

    generator = None
    if os.environ.get("DEXGRASP_SAMPLER_SEED"):
        generator = torch.Generator()
        generator.manual_seed(int(os.environ["DEXGRASP_SAMPLER_SEED"]))

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )

    summary = []
    for obj_id, count in zip(obj_ids, obj_counts):
        obj_id = int(obj_id)
        obj_name = dataset.obj_code_name_list[obj_id]
        summary.append(
            {
                "obj_code_idx": obj_id,
                "obj_code": obj_name,
                "frame_count": int(count),
                "original_prob": float(count / len(frame_obj_ids)),
                "sample_weight": float(1.0 / count),
                "expected_epoch_samples": float(len(frame_obj_ids) / len(obj_ids)),
            }
        )

    return sampler, summary


class CheckpointLossLogger(pl.Callback):
    def __init__(self, exp_dir):
        super().__init__()
        self.exp_dir = exp_dir
        self.csv_path = os.path.join(exp_dir, "checkpoint_loss.csv")
        self.png_path = os.path.join(exp_dir, "checkpoint_loss_curve.png")
        self.rows = []
        self._loaded = False

    @staticmethod
    def _metric_to_float(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.detach().cpu().reshape(-1)[0].item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _load_existing(self):
        if self._loaded:
            return
        self._loaded = True
        if not os.path.exists(self.csv_path):
            return
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = {}
                for key, value in row.items():
                    if key in {"epoch", "global_step"}:
                        parsed[key] = int(float(value)) if value not in (None, "") else -1
                    elif key == "source":
                        parsed[key] = value
                    else:
                        parsed[key] = float(value) if value not in (None, "") else None
                self.rows.append(parsed)

    def _current_row(self, trainer, source):
        metrics = trainer.callback_metrics
        metric_names = [
            "train_loss",
            "val_loss",
            "wrist_loss",
            "ori_loss",
            "finger_loss",
            "l1_loss",
            "val_wrist_loss",
            "val_ori_loss",
            "val_finger_loss",
            "val_l1_loss",
        ]
        row = {
            "epoch": int(getattr(trainer, "current_epoch", -1)),
            "global_step": int(getattr(trainer, "global_step", -1)),
            "source": source,
        }
        for name in metric_names:
            row[name] = self._metric_to_float(metrics.get(name))
        return row

    def record(self, trainer, source):
        self._load_existing()
        row = self._current_row(trainer, source)
        if self.rows:
            last = self.rows[-1]
            if (
                last.get("epoch") == row.get("epoch")
                and last.get("global_step") == row.get("global_step")
                and last.get("source") == row.get("source")
            ):
                return
        self.rows.append(row)
        self.write_csv()
        # print(
        #     "checkpoint_loss_log: source={} epoch={} step={} train_loss={} val_loss={}".format(
        #         row["source"],
        #         row["epoch"],
        #         row["global_step"],
        #         "None" if row["train_loss"] is None else "{:.8f}".format(row["train_loss"]),
        #         "None" if row["val_loss"] is None else "{:.8f}".format(row["val_loss"]),
        #     )
        # )

    def write_csv(self):
        if not self.rows:
            return
        os.makedirs(self.exp_dir, exist_ok=True)
        fieldnames = [
            "epoch",
            "global_step",
            "source",
            "train_loss",
            "val_loss",
            "wrist_loss",
            "ori_loss",
            "finger_loss",
            "l1_loss",
            "val_wrist_loss",
            "val_ori_loss",
            "val_finger_loss",
            "val_l1_loss",
        ]
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)

    def plot(self):
        self._load_existing()
        if not self.rows:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            print("checkpoint_loss_plot skipped: matplotlib import failed: {}".format(exc))
            return

        plot_rows = [
            row for row in self.rows
            if row.get("train_loss") is not None or row.get("val_loss") is not None
        ]
        if not plot_rows:
            return

        x = [row["epoch"] for row in plot_rows]
        train_loss = [row.get("train_loss") for row in plot_rows]
        val_loss = [row.get("val_loss") for row in plot_rows]

        fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
        if any(value is not None for value in train_loss):
            ax.plot(x, train_loss, marker="o", linewidth=1.6, label="train_loss")
        if any(value is not None for value in val_loss):
            ax.plot(x, val_loss, marker="s", linewidth=1.3, label="val_loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Checkpoint Loss Curve")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.png_path)
        plt.close(fig)
        print("checkpoint_loss_plot saved: {}".format(self.png_path))

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self.record(trainer, source="validation")

    def on_train_end(self, trainer, pl_module):
        self.plot()


class BCTrainer:
    def __init__(self, args, env_args, train_loader=None, test_loader=None):
        self.args = args
        self.env_args = env_args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.bc_model = LitBCModel(args, env_args.env)

    def train(self, ckpt_path=None):
        ckpt_every = _runtime_int(self.args, "ckpt_every_n_epochs", "DEXGRASP_CKPT_EVERY_N_EPOCHS")
        save_best = _runtime_flag(
            self.args,
            "save_best_ckpt",
            "DEXGRASP_SAVE_BEST_CKPT",
            default=True,
        )
        best_monitor = _runtime_str(
            self.args,
            "best_monitor",
            "DEXGRASP_BEST_MONITOR",
            "val_loss",
        )
        best_mode = _runtime_str(
            self.args,
            "best_mode",
            "DEXGRASP_BEST_MODE",
            "min",
        )
        if best_mode not in {"min", "max"}:
            raise ValueError("best_mode must be 'min' or 'max', got {}".format(best_mode))

        checkpoint_callbacks = []
        best_callback = None
        if save_best:
            best_callback = ModelCheckpoint(
                dirpath=self.args.exp_dir,
                filename="best-epoch={epoch:03d}-step={step}",
                auto_insert_metric_name=False,
                monitor=best_monitor,
                mode=best_mode,
                save_top_k=1,
                save_last=False,
                save_on_train_epoch_end=False,
            )
            checkpoint_callbacks.append(best_callback)

        last_callback = ModelCheckpoint(
            dirpath=self.args.exp_dir,
            filename="last",
            auto_insert_metric_name=False,
            monitor=None,
            save_top_k=0,
            save_last=True,
            every_n_epochs=1,
            save_on_train_epoch_end=True,
        )
        checkpoint_callbacks.append(last_callback)

        if _runtime_flag(self.args, "save_epoch_ckpt", "DEXGRASP_SAVE_EPOCH_CKPT"):
            periodic_every = 1 if ckpt_every is None else ckpt_every
            if periodic_every <= 0:
                raise ValueError("ckpt_every_n_epochs must be positive")
            periodic_callback = ModelCheckpoint(
                dirpath=self.args.exp_dir,
                filename="epoch={epoch:03d}-step={step}",
                auto_insert_metric_name=False,
                save_top_k=-1,
                save_last=False,
                every_n_epochs=periodic_every,
                save_on_train_epoch_end=True,
            )
            checkpoint_callbacks.append(periodic_callback)

        lr_monitor = LearningRateMonitor(logging_interval="step")
        loss_logger = CheckpointLossLogger(self.args.exp_dir)
        check_val_every = _runtime_int(
            self.args,
            "check_val_every_n_epoch",
            "DEXGRASP_CHECK_VAL_EVERY_N_EPOCH",
            1,
        )
        if check_val_every <= 0:
            raise ValueError("check_val_every_n_epoch must be positive")
        deterministic = _runtime_flag(
            self.args,
            "deterministic",
            "DEXGRASP_DETERMINISTIC",
            default=True,
        )
        print(
            "checkpoint_config save_best={} monitor={} mode={} save_last=1 "
            "save_periodic={} periodic_every={} check_val_every={}".format(
                int(save_best),
                best_monitor,
                best_mode,
                int(_runtime_flag(self.args, "save_epoch_ckpt", "DEXGRASP_SAVE_EPOCH_CKPT")),
                "disabled" if ckpt_every is None else ckpt_every,
                check_val_every,
            )
        )
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            precision=32,
            max_epochs=self.args.num_epochs,
            callbacks=checkpoint_callbacks + [lr_monitor, loss_logger],
            log_every_n_steps=5,
            check_val_every_n_epoch=check_val_every,
            deterministic=deterministic,
            default_root_dir=os.path.join(self.args.exp_dir, "tensorboard_logs"),
        )

        try:
            trainer.fit(
                model=self.bc_model,
                train_dataloaders=self.train_loader,
                ckpt_path=ckpt_path,
                val_dataloaders=self.test_loader,
            )
        except KeyboardInterrupt:
            last_path = os.path.join(self.args.exp_dir, "last.ckpt")
            trainer.save_checkpoint(last_path)
            loss_logger.record(trainer, source="interrupted")
            loss_logger.plot()
            print("training interrupted: saved checkpoint {}".format(last_path))
            raise
        except Exception:
            loss_logger.plot()
            raise
        finally:
            loss_logger.write_csv()

        if _runtime_flag(self.args, "save_final_ckpt", "DEXGRASP_SAVE_FINAL_CKPT"):
            last_path = last_callback.last_model_path or os.path.join(self.args.exp_dir, "last.ckpt")
            if not os.path.exists(last_path):
                trainer.save_checkpoint(last_path)
            loss_logger.record(trainer, source="final")
            loss_logger.plot()

        best_path = ""
        best_score = None
        if best_callback is not None:
            best_path = best_callback.best_model_path
            best_score = CheckpointLossLogger._metric_to_float(best_callback.best_model_score)
            if not best_path or not os.path.exists(best_path):
                raise RuntimeError(
                    "best checkpoint was requested but no checkpoint was saved for {}".format(
                        best_monitor
                    )
                )

        last_path = last_callback.last_model_path or os.path.join(self.args.exp_dir, "last.ckpt")
        selection_manifest = {
            "best": {
                "path": os.path.abspath(best_path) if best_path else None,
                "monitor": best_monitor if best_callback is not None else None,
                "mode": best_mode if best_callback is not None else None,
                "score": best_score,
            },
            "last": {
                "path": os.path.abspath(last_path),
                "completed_epochs": int(self.args.num_epochs),
                "global_step": int(trainer.global_step),
            },
        }
        manifest_path = os.path.join(self.args.exp_dir, "checkpoint_selection.yaml")
        with open(manifest_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(selection_manifest, f, sort_keys=False)
        print("best_checkpoint path={} score={}".format(best_path or "disabled", best_score))
        print("last_checkpoint path={}".format(last_path))
        print("checkpoint_selection_manifest path={}".format(manifest_path))


def main(args, env_args):
    train_seed = _runtime_int(args, "train_seed", "DEXGRASP_TRAIN_SEED")
    if train_seed is not None:
        pl.seed_everything(train_seed, workers=True)
        print("train_seed={}".format(train_seed))

    kstr = "sim_action" if args.use_sim_action else "vis_action"
    args.task_name = "1obj_seq2000_DexRep_pro{}_start_uniform_{}_dsam_mod".format(args.obs_dim, kstr)

    exp_suffix = _runtime_str(args, "exp_suffix", "DEXGRASP_EXP_SUFFIX")
    if exp_suffix:
        args.task_name = "{}_{}".format(args.task_name, exp_suffix)
    num_epochs = _runtime_int(args, "num_epochs", "DEXGRASP_NUM_EPOCHS")
    if num_epochs is not None:
        args.num_epochs = num_epochs
    batch_size = _runtime_int(args, "batch_size", "DEXGRASP_BATCH_SIZE")
    if batch_size is not None:
        args.batch_size = batch_size
    train_data_dir = _runtime_str(args, "train_data_dir", "DEXGRASP_TRAIN_DATA_DIR")
    if train_data_dir:
        args.train_data_dir = train_data_dir
    test_data_dir = _runtime_str(args, "test_data_dir", "DEXGRASP_TEST_DATA_DIR")
    if test_data_dir:
        args.test_data_dir = test_data_dir
    seq_num = _runtime_int(args, "seq_num", "DEXGRASP_SEQ_NUM")
    if seq_num is not None:
        args.seq_num = seq_num
    val_seq_num = _runtime_int(args, "val_seq_num", "DEXGRASP_VAL_SEQ_NUM")
    if val_seq_num is not None:
        args.val_seq_num = val_seq_num
    obs_dim = _runtime_int(args, "obs_dim", "DEXGRASP_OBS_DIM")
    if obs_dim is not None:
        args.obs_dim = obs_dim
    o6_obs_mode = _runtime_str(args, "o6_obs_mode", "DEXGRASP_O6_OBS_MODE")
    if o6_obs_mode:
        args.o6_obs_mode = o6_obs_mode
    dexrep_sensor_dim = _runtime_int(args, "dexrep_sensor_dim", "DEXGRASP_DEXREP_SENSOR_DIM")
    if dexrep_sensor_dim is not None:
        args.dexrep_sensor_dim = dexrep_sensor_dim
    dexrep_pnl_dim = _runtime_int(args, "dexrep_pnl_dim", "DEXGRASP_DEXREP_PNL_DIM")
    if dexrep_pnl_dim is not None:
        args.dexrep_pnl_dim = dexrep_pnl_dim
    train_obj_codes = _runtime_list(args, "train_obj_code_list", "DEXGRASP_TRAIN_OBJ_CODES")
    if train_obj_codes is not None:
        args.train_obj_code_list = train_obj_codes

    args.policy.actor_critic = _runtime_str(
        args,
        "policy_actor_critic",
        "DEXGRASP_POLICY_ACTOR_CRITIC",
        args.policy.actor_critic,
    )
    temporal_history = _runtime_int(args, "temporal_history", "DEXGRASP_TEMPORAL_HISTORY")
    if temporal_history is not None:
        args.encoder.n_obs_steps = temporal_history
    tcn_dropout = _runtime_float(args, "tcn_dropout", "DEXGRASP_TCN_DROPOUT")
    if tcn_dropout is not None:
        args.encoder.tcn_dropout = tcn_dropout
    tcn_kernel_size = _runtime_int(args, "tcn_kernel_size", "DEXGRASP_TCN_KERNEL_SIZE")
    if tcn_kernel_size is not None:
        args.encoder.tcn_kernel_size = tcn_kernel_size
    tcn_dilations = _runtime_list(args, "tcn_dilations", "DEXGRASP_TCN_DILATIONS")
    if tcn_dilations is not None:
        args.encoder.tcn_dilations = [int(value) for value in tcn_dilations]

    if args.policy.actor_critic == "ActorCriticPNG":
        env_args.env.obs_dim.pop("dexrep_sensor")
        env_args.env.obs_dim.pop("dexrep_pnl")
    else:
        configure_o6_dexrep_obs_dims(args, env_args)
        env_args.env.obs_dim.pop("pnG")

    args.add_noise = False
    args.noise_val = 0.02
    args.exp_dir = os.path.join(args.exp_dir, args.task_name)
    os.makedirs(args.exp_dir, exist_ok=True)

    print("train_config task_name={}".format(args.task_name))
    print("train_config actor_critic={}".format(args.policy.actor_critic))
    print("train_config obs_type={} obs_dim={}".format(args.obs_type, args.obs_dim))
    print("train_config temporal_history={}".format(args.encoder.n_obs_steps))
    print("train_config tcn_dropout={}".format(getattr(args.encoder, "tcn_dropout", "unset")))
    print("train_config tcn_kernel_size={}".format(getattr(args.encoder, "tcn_kernel_size", "unset")))
    print("train_config tcn_dilations={}".format(getattr(args.encoder, "tcn_dilations", "unset")))
    object_balanced_sampler = _runtime_flag(args, "object_balanced_sampler", "DEXGRASP_OBJECT_BALANCED_SAMPLER")
    sampler_seed = _runtime_int(args, "sampler_seed", "DEXGRASP_SAMPLER_SEED")
    ckpt_every = _runtime_int(args, "ckpt_every_n_epochs", "DEXGRASP_CKPT_EVERY_N_EPOCHS")
    num_workers = _runtime_int(args, "num_workers", "DEXGRASP_NUM_WORKERS", 4)
    print("train_config object_balanced_sampler={}".format(int(object_balanced_sampler)))
    print("train_config sampler_seed={}".format("unset" if sampler_seed is None else sampler_seed))
    print("train_config num_epochs={} ckpt_every={}".format(args.num_epochs, "default" if ckpt_every is None else ckpt_every))
    print("train_config batch_size={} num_workers={}".format(args.batch_size, num_workers))
    print("train_config train_data_dir={}".format(args.train_data_dir))
    print("train_config test_data_dir={}".format(args.test_data_dir))
    print("train_config train_obj_codes={}".format(args.train_obj_code_list))
    print("train_config exp_dir={}".format(args.exp_dir))

    ds_train = GraspM3DexRepDataset(args, ds_name="train")
    ds_test = GraspM3DexRepDataset(args, ds_name="test")
    configure_o6_dexrep_obs_dims(args, env_args, ds_train)
    test_obs_dim = int(ds_test.data["obs"].shape[-1])
    train_obs_dim = int(ds_train.data["obs"].shape[-1])
    if test_obs_dim != train_obs_dim:
        raise ValueError("train/test obs dim mismatch: {} vs {}".format(train_obs_dim, test_obs_dim))
    print("train_config effective_obs_dim={}".format(train_obs_dim))
    print("train_config effective_env_obs_dim={}".format(dict(env_args.env.obs_dim)))

    if object_balanced_sampler:
        if sampler_seed is not None:
            os.environ["DEXGRASP_SAMPLER_SEED"] = str(sampler_seed)
        train_sampler, sampler_summary = build_object_balanced_sampler(ds_train)
        print("object_balanced_sampler enabled")
        print("object_balanced_sampler num_samples={}".format(len(ds_train)))
        for row in sampler_summary:
            print(
                "object_balance obj={obj_code} frame_count={frame_count} "
                "original_prob={original_prob:.6f} sample_weight={sample_weight:.10f} "
                "expected_epoch_samples={expected_epoch_samples:.2f}".format(**row)
            )
        train_loader = DataLoader(
            ds_train,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=False,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            ds_train,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
        )
    test_loader = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    trainer = BCTrainer(args, env_args, train_loader, test_loader)
    resume_ckpt = _runtime_str(args, "resume_ckpt", "DEXGRASP_RESUME_CKPT")
    print("train_config resume_ckpt={}".format(resume_ckpt or "None"))
    trainer.train(ckpt_path=resume_ckpt)


if __name__ == "__main__":
    assert_clean_runtime_paths()
    from omegaconf import OmegaConf

    bc_config_name = os.environ.get("DEXGRASP_BC_CONFIG", "lhm_bc_o6_dexrep_full.yaml")
    if not bc_config_name.endswith(".yaml"):
        bc_config_name = "{}.yaml".format(bc_config_name)
    args = OmegaConf.load("{}/{}".format("../ActionDiffusion/bc/config", bc_config_name))
    env_args = OmegaConf.load("{}/o6_hand_grasp_dexrep_ijrr.yaml".format("./cfg"))
    main(args, env_args)
