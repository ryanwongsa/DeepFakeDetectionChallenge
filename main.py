from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.lightning_system import LightningSystem

model = LightningSystem()

trainer = Trainer(gpus=1, nb_sanity_val_steps=0,overfit_pct=0.01)
trainer.fit(model)   