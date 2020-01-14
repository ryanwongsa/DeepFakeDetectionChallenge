from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.lightning_system import LightningSystem
import os

model = LightningSystem()

checkpoint_callback = ModelCheckpoint(
    filepath=os.getcwd(),
    save_best_only=True,
    verbose=True,
    monitor='loss',
    mode='min',
    prefix=''
)

trainer = Trainer(gpus=1, nb_sanity_val_steps=0,overfit_pct=0.01, checkpoint_callback=checkpoint_callback)
trainer.fit(model)   