import pytorch_lightning as pl
import torch
import torchmetrics
import numpy as np
from genai_project.model.imagen.frozen_text_encoder import FrozenTextEncoder


class Imagen(pl.LightningModule):
    def __init__(self):
        super(Imagen, self).__init__()
        # TODO

        self._text_encoder = FrozenTextEncoder()

        raise NotImplementedError("Imagen model is not implemented yet.")
        self.save_hyperparameters()

    def forward(self, x):
        raise NotImplementedError("Forward pass is not implemented yet.")

    def configure_optimizers(self):
        raise NotImplementedError("Optimizer configuration is not implemented yet.")

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Training step is not implemented yet.")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("Validation step is not implemented yet.")
    
    def test_step(self, batch, batch_idx):
        raise NotImplementedError("Test step is not implemented yet.")
