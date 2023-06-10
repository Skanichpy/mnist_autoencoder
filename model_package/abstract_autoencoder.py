from torch import optim
import pytorch_lightning as pl

from typing import Callable, Any
from abc import abstractmethod

class AbstractAutoEncoder(pl.LightningModule): 

    def __init__(self, loss:Callable, lr:float):
        super().__init__()
        self.loss = loss 
        self.lr = lr 

    def forward(self, x): 
        context = self.encoder(x)
        out = self.decoder(context)
        return out
    
    def validation_step(self, batch, batch_idx):
        x, gt, _ = batch
        val_batch_loss = self.loss(self(x), gt)
        self.log(name="VAL_LOSS", value=val_batch_loss,
                 prog_bar=True, on_epoch=True)
    
    def training_step(self, batch, batch_idx):
        x, gt, _ = batch 
        out = self(x)
        train_batch_loss = self.loss(out, gt)
        self.log(name="LOSS", value=train_batch_loss, prog_bar=True,
                 on_epoch=True, on_step=True)
        return train_batch_loss
    
    def predict_step(self, batch, batch_idx):
        self.reduce_dim(batch, batch_idx)
    
    def reduce_dim(self, batch, batch_idx):
        x, _, _ = batch 
        return self.encoder(x)
    
    def configure_optimizers(self):
        return optim.Adam([*self.encoder.parameters(),
                           *self.decoder.parameters()], lr=self.lr,
                           weight_decay=1e-5)
    @abstractmethod
    def __build_encoder(self, **kwargs): 
        raise NotImplementedError(f'{self} must have build_encoder method')
    
    @abstractmethod
    def __build_decoder(self, **kwargs):
        raise NotImplementedError(f'{self} must have build_decoder method')