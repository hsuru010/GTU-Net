
# +---------------------------------------------------------------------------------------+ #
# |                                                                                       | #
# |                               Main Lightning Module                                   | #
# |                                                                                       | #
# +---------------------------------------------------------------------------------------+ #

import torch
from torch import nn
import torch.optim as optim
import lightning as L
import torch.nn.functional as F



#This is where the main model training process occurs using a Lightning Module.
#Here, we do our training step, validation step, testing step, and final predict step.
#We also load our models, configure the optimizers, and compute the loss.




class ADT_Lightning_Module(L.LightningModule):
    def __init__(self, models_dict, model_name, **kwargs):
        super().__init__()
        """
        Main training class.
        Includes optimizers, loss function, training step, validation step, testing step, prediction step.

        Args
        ----
        models_dict: dict
            dictionary with the number of channels and number of classes for each model
        model_name: str
            name of the model
        **kwargs: str
            model parameters found in the models_dict
        """

        self.model_name = model_name
        self.model = models_dict[model_name](**kwargs)

        self.loss_criterion = nn.MSELoss(reduction='none')

    def forward(self, x1, x2=None):


        if self.model_name == "smaat_unet_adt":
            return self.model(x1)
        elif self.model_name == "smaat_unet_sst":
            return self.model(x1, x2)
        elif self.model_name == "smaat_unet_adt_sst":
            return self.model(x1, x2)
        else:
            raise ValueError("WRONG MODEL NAME INPUT")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.0001, weight_decay=0.01)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                    factor=0.5, patience=3,
                                                                    verbose=True),
            'monitor': 'val_loss'
        }
        return [optimizer], [lr_scheduler]

    def compute_loss(self, predictions, target_sequence, f=1e-4, alpha=0.8, beta=0.5, lambda_geo = 5e-9 ):
        """
        Composite loss for SSH prediction:
        1. MSE + MAE (data-driven)
        2. Two-Step temporal consistency loss
        3. Geostrophic balance loss (ugos, vgos)

        Args:
            predictions: (B, T, H, W)
            target_sequence: (B, T, H, W)
            f: Coriolis parameter (default 1e-4)
            alpha, beta, lambda_geo: weighting factors
        """
        g = 9.81

        # ------------------ 1️⃣ Data-driven Loss ------------------
        mse_loss = F.mse_loss(predictions, target_sequence)
        mae_loss = F.l1_loss(predictions, target_sequence)
        basic_loss = alpha * mse_loss + (1 - alpha) * mae_loss

        # ------------------ 2️⃣ Two-Step Temporal Loss ------------------
        if predictions.size(1) > 1:  # T > 1
            next_step_pred = predictions[:, :-1, :, :]
            target_next_step = target_sequence[:, 1:, :, :]
            two_step_loss = F.mse_loss(next_step_pred, target_next_step)
        else:
            two_step_loss = torch.tensor(0.0, device=predictions.device)

        # ------------------ 3️⃣ Geostrophic Balance Loss ------------------
        # 计算梯度 (空间差分)
        d_eta_dy = predictions[:, :, 1:, :] - predictions[:, :, :-1, :]  # dy: south-north
        d_eta_dx = predictions[:, :, :, 1:] - predictions[:, :, :, :-1]  # dx: west-east

        d_eta_dy_target = target_sequence[:, :, 1:, :] - target_sequence[:, :, :-1, :]
        d_eta_dx_target = target_sequence[:, :, :, 1:] - target_sequence[:, :, :, :-1]

        # 计算地转速度 (ugos, vgos)
        u_geo_pred = -g / f * d_eta_dy
        v_geo_pred = g / f * d_eta_dx
        u_geo_true = -g / f * d_eta_dy_target
        v_geo_true = g / f * d_eta_dx_target

        geo_loss = F.mse_loss(u_geo_pred, u_geo_true) + F.mse_loss(v_geo_pred, v_geo_true)

        # ------------------ 4️⃣ Final Weighted Loss ------------------
        total_loss = basic_loss + beta * two_step_loss + lambda_geo * geo_loss
        return total_loss


    def training_step(self, batch, batch_idx):
        if self.model_name == 'smaat_unet_adt':
            input_sequences, target_sequences = batch
            predictions = self(input_sequences)
            loss = self.compute_loss(predictions, target_sequences,alpha=0.8, beta=0.5)
        else:
            input_adt, input_sst, target_adt, target_sst = batch['input_adt'], batch['input_sst'], batch['target_adt'], batch['target_sst']
            if self.model_name == 'smaat_unet_adt_sst':
                predictions_adt, predictions_sst = self(input_adt, input_sst)

                loss_adt = self.compute_loss(predictions_adt, target_adt,alpha=0.8, beta=0.5)
                loss_sst = self.compute_loss(predictions_sst, target_sst,alpha=0.8, beta=0.5)
                loss = loss_adt + loss_sst
            elif self.model_name == 'smaat_unet_sst':
                predictions = self(input_adt, input_sst)
                loss = self.compute_loss(predictions, target_adt,alpha=0.8, beta=0.5)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.model_name == 'smaat_unet_adt':
            input_sequences, target_sequences = batch
            predictions = self(input_sequences)
            loss = self.compute_loss(predictions, target_sequences,alpha=0.8, beta=0.5)
        else:
            input_adt, input_sst, target_adt, target_sst = batch['input_adt'], batch['input_sst'], batch['target_adt'], batch['target_sst']
            if self.model_name == 'smaat_unet_adt_sst':
                predictions_adt, predictions_sst = self(input_adt, input_sst)
                loss_adt = self.compute_loss(predictions_adt, target_adt,alpha=0.8, beta=0.5)
                loss_sst = self.compute_loss(predictions_sst, target_sst,alpha=0.8, beta=0.5)
                loss = loss_adt + loss_sst
            elif self.model_name == 'smaat_unet_sst':
                predictions = self(input_adt, input_sst)
                loss = self.compute_loss(predictions, target_adt,alpha=0.8, beta=0.5)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        if self.model_name == 'smaat_unet_adt':
            input_sequences, target_sequences = batch
            predictions = self(input_sequences)
            loss = self.compute_loss(predictions, target_sequences,alpha=0.8, beta=0.5)
        else:
            input_adt, input_sst, target_adt, target_sst = batch['input_adt'], batch['input_sst'], batch['target_adt'], batch['target_sst']
            if self.model_name == 'smaat_unet_adt_sst':
                predictions_adt, predictions_sst = self(input_adt, input_sst)
                loss_adt = self.compute_loss(predictions_adt, target_adt,alpha=0.8, beta=0.5)
                loss_sst = self.compute_loss(predictions_sst, target_sst,alpha=0.8, beta=0.5)
                loss = loss_adt + loss_sst
            elif self.model_name == 'smaat_unet_sst':
                predictions = self(input_adt, input_sst)
                loss = self.compute_loss(predictions, target_adt,alpha=0.8, beta=0.5)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        if self.model_name == 'smaat_unet_adt':
            input_sequences, _ = batch
            predictions = self(input_sequences)
        else:
            input_adt, input_sst, _, _ = batch['input_adt'], batch['input_sst'], batch['target_adt'], batch['target_sst']
            if self.model_name == 'smaat_unet_adt_sst':
                predictions_adt, predictions_sst = self(input_adt, input_sst)
                predictions = (predictions_adt, predictions_sst)
            elif self.model_name == 'smaat_unet_sst':
                predictions = self(input_adt, input_sst)

        return predictions