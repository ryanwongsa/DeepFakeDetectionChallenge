
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.lightning_system import LightningSystem

model = LightningSystem()

# # DEFAULTS used by the Trainer
checkpoint_callback = ModelCheckpoint(
    filepath="/dltraining/checkpoints/test/",
    save_best_only=False,
    verbose=True,
    monitor='val_loss',
    mode='min',
    prefix=''
)

trainer = Trainer(nb_sanity_val_steps=0, gpus=1, max_nb_epochs=1, train_percent_check=0.1, val_percent_check=1.0, checkpoint_callback=checkpoint_callback) 

trainer.fit(model)  