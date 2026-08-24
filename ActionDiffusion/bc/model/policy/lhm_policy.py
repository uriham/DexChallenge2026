import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from ActionDiffusion.bc.model.policy.lqt_policy import (
    ActorCriticDexRep,
    ActorCriticDexRepTemporal,
    ActorCriticDexRepTemporalTCN,
    ActorCriticPNG,
)


class LitBCModel(pl.LightningModule):
    def __init__(self, args, env_args):
        super().__init__()
        self.args = args

        self.model_cfg = self.args.policy
        self.actions_shape = self.model_cfg.actions_shape
        self.learn_cfg = self.args.learn
        self.init_noise_std = self.learn_cfg.get("init_noise_std", 0.3)
        self.encoder_cfg = self.args.encoder
        self.cfg_env = env_args
        self.model = eval(self.model_cfg.actor_critic)(
            None,
            self.actions_shape,
            self.init_noise_std,
            self.model_cfg,
            self.encoder_cfg,
            self.cfg_env,
        )

    def forward(self, batch):
        actions_mean = self.model(batch['obs'])
        return actions_mean

    def training_step(self, batch, batch_idx):
        pred_action = self.forward(batch)
        loss_dct = self.cal_loss(pred_action, batch['actions'])

        self.log_dict({'train_loss': loss_dct['loss']}, prog_bar=True, on_epoch=True)
        self.log_dict({'wrist_loss': loss_dct['wrist_loss']}, prog_bar=True, on_epoch=True)
        self.log_dict({'ori_loss': loss_dct['ori_loss']}, prog_bar=True, on_epoch=True)
        self.log_dict({'finger_loss': loss_dct['finger_loss']}, prog_bar=True, on_epoch=True)
        self.log_dict({'l1_loss': loss_dct['l1_loss']}, prog_bar=True, on_epoch=True)

        return loss_dct['loss']

    def validation_step(self, batch, batch_idx):
        pred_action = self.forward(batch)
        loss_dct = self.cal_loss(pred_action, batch['actions'])

        batch_size = int(batch['actions'].shape[0])
        self.log(
            'val_loss',
            loss_dct['loss'],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log_dict(
            {
                'val_wrist_loss': loss_dct['wrist_loss'],
                'val_ori_loss': loss_dct['ori_loss'],
                'val_finger_loss': loss_dct['finger_loss'],
                'val_l1_loss': loss_dct['l1_loss'],
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

    def cal_loss(self, predict, gt):
        wrist_loss = F.mse_loss(predict[:, :3], gt[:, :3])
        ori_loss = F.mse_loss(predict[:, 3:6], gt[:, 3:6])
        finger_loss = F.mse_loss(predict[:, 6:], gt[:, 6:])
        l1_loss = F.l1_loss(predict, gt)

        loss = 2 * wrist_loss + ori_loss + finger_loss + l1_loss

        loss_dict = {
            "loss": loss,
            'wrist_loss': wrist_loss,
            'ori_loss': ori_loss,
            'finger_loss': finger_loss,
            'l1_loss': l1_loss,
        }

        return loss_dict

    def configure_optimizers(self):
        params_list = [{'params': self.parameters(), 'lr': self.args.lr}]
        optimizer = torch.optim.Adam(params_list)
        return {"optimizer": optimizer}
