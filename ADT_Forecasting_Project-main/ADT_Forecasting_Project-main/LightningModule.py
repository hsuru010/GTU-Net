
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




class SLA_Lightning_Module(L.LightningModule):
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
        """
        Args
        ----
        x1: tensor
            SLA data tensor input for the network
        x2: tensor, optional
            SST data tensor input for the network. The default is None unless using SST data.
        """

        if self.model_name == "smaat_unet_sla":
            return self.model(x1)
        elif self.model_name == "smaat_unet_sst":
            return self.model(x1, x2)
        elif self.model_name == "smaat_unet_sla_sst":
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




    def compute_loss(self, predictions, target_sequence,  alpha=0.8, beta=0.5):
        """
        Improved loss function with Two-Step Loss for better long-term forecasting.
        - alpha: Balances MSE and MAE loss.
        - beta: Weight for two-step loss component.
        """
        mse_loss = self.loss_criterion(predictions, target_sequence)  # Single-step loss

        sequence_length = target_sequence.size(1)

        loss_mse = mse_loss.mean()

        # Add MAE loss to improve long-term stability
        loss_mae = F.l1_loss(predictions, target_sequence)

        # ------------------ Two-Step Loss ------------------
        # Simulate one-step prediction
        next_step_pred = predictions[:, :-1]  # Remove last time step
        target_next_step = target_sequence[:, 1:]  # Shift target forward

        # Compute second-step prediction loss
        loss_two_step = F.mse_loss(next_step_pred, target_next_step)

        # Final loss combining single-step, two-step, and MAE
        loss = alpha * loss_mse + (1 - alpha) * loss_mae + beta * loss_two_step

        return loss

    def training_step(self, batch, batch_idx):
        if self.model_name == 'smaat_unet_sla':
            input_sequences, target_sequences = batch
            predictions = self(input_sequences)
            loss = self.compute_loss(predictions, target_sequences,alpha=0.8, beta=0.5)
        else:
            input_sla, input_sst, target_sla, target_sst = batch['input_sla'], batch['input_sst'], batch['target_sla'], batch['target_sst']
            if self.model_name == 'smaat_unet_sla_sst':
                predictions_sla, predictions_sst = self(input_sla, input_sst)

                loss_sla = self.compute_loss(predictions_sla, target_sla,alpha=0.8, beta=0.5)
                loss_sst = self.compute_loss(predictions_sst, target_sst,alpha=0.8, beta=0.5)
                loss = loss_sla + loss_sst
            elif self.model_name == 'smaat_unet_sst':
                predictions = self(input_sla, input_sst)
                loss = self.compute_loss(predictions, target_sla,alpha=0.8, beta=0.5)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.model_name == 'smaat_unet_sla':
            input_sequences, target_sequences = batch
            predictions = self(input_sequences)
            loss = self.compute_loss(predictions, target_sequences,alpha=0.8, beta=0.5)
        else:
            input_sla, input_sst, target_sla, target_sst = batch['input_sla'], batch['input_sst'], batch['target_sla'], batch['target_sst']
            if self.model_name == 'smaat_unet_sla_sst':
                predictions_sla, predictions_sst = self(input_sla, input_sst)
                loss_sla = self.compute_loss(predictions_sla, target_sla,alpha=0.8, beta=0.5)
                loss_sst = self.compute_loss(predictions_sst, target_sst,alpha=0.8, beta=0.5)
                loss = loss_sla + loss_sst
            elif self.model_name == 'smaat_unet_sst':
                predictions = self(input_sla, input_sst)
                loss = self.compute_loss(predictions, target_sla,alpha=0.8, beta=0.5)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        if self.model_name == 'smaat_unet_sla':
            input_sequences, target_sequences = batch
            predictions = self(input_sequences)
            loss = self.compute_loss(predictions, target_sequences,alpha=0.8, beta=0.5)
        else:
            input_sla, input_sst, target_sla, target_sst = batch['input_sla'], batch['input_sst'], batch['target_sla'], batch['target_sst']
            if self.model_name == 'smaat_unet_sla_sst':
                predictions_sla, predictions_sst = self(input_sla, input_sst)
                loss_sla = self.compute_loss(predictions_sla, target_sla,alpha=0.8, beta=0.5)
                loss_sst = self.compute_loss(predictions_sst, target_sst,alpha=0.8, beta=0.5)
                loss = loss_sla + loss_sst
            elif self.model_name == 'smaat_unet_sst':
                predictions = self(input_sla, input_sst)
                loss = self.compute_loss(predictions, target_sla,alpha=0.8, beta=0.5)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx):
        if self.model_name == 'smaat_unet_sla':
            input_sequences, _ = batch
            predictions = self(input_sequences)
        else:
            input_sla, input_sst, _, _ = batch['input_sla'], batch['input_sst'], batch['target_sla'], batch['target_sst']
            if self.model_name == 'smaat_unet_sla_sst':
                predictions_sla, predictions_sst = self(input_sla, input_sst)
                predictions = (predictions_sla, predictions_sst)
            elif self.model_name == 'smaat_unet_sst':
                predictions = self(input_sla, input_sst)

        return predictions