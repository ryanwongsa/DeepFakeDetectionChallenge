from lightning.lightning_system import LightningSystem
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
from pytorch_lightning.logging import TestTubeLogger

import os
import string
import random

def main(hparams):
    
    logger = TestTubeLogger(
        save_dir=hparams.save_path,
        version=hparams.version_number 
    )
    
    if hparams.checkpoint_dir is not None:
        model = LightningSystem.load_from_checkpoint(
            checkpoint_path=hparams.checkpoint_dir, resume_run=hparams.resume_run
        )
        print("Loaded from pretrained model: ", model.hparams.resume_run)
    else:
        model = LightningSystem(hparams)

    checkpoint_callback = ModelCheckpoint(
        filepath=hparams.save_path,
        save_best_only=False,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=hparams.checkpoint_name
    )

    trainer = Trainer(
        nb_sanity_val_steps=0, 
        gpus=hparams.num_gpus,
        logger=logger,
        train_percent_check=hparams.train_percent_check, 
        val_check_interval=hparams.val_check_interval,
        val_percent_check=hparams.val_percent_check,
        use_amp=hparams.use_16bit,
        default_save_path=hparams.save_path,
        checkpoint_callback=checkpoint_callback,
        max_nb_epochs = hparams.num_epochs
    )

    print("Checkpoint Prefix:", hparams.checkpoint_name)
    trainer.fit(model)

if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)
    checkpoint_prefix = ''.join(random.choices(string.ascii_letters, k=8))


    parent_parser.add_argument(
        '--train_percent_check',
        type=float,
        default=1.0,
    )

    parent_parser.add_argument(
        '--val_check_interval',
        type=float,
        default=1.0,
    )

    parent_parser.add_argument(
        '--val_percent_check',
        type=float,
        default=1.0,
    )

    parent_parser.add_argument(
        '--num_epochs',
        type=int,
        default=3,
    )

    parent_parser.add_argument(
        '--checkpoint_dir',
        type=str
    )

    parent_parser.add_argument(
        '--version_number',
        type=int,
        default=None
    )

    parent_parser.add_argument(
        '--save_path',
        type=str
    )
    
    parent_parser.add_argument(
        '--resume_run',
       default=None,
       type=str
    )
    
    parent_parser.add_argument(
        '--use_16bit',
        dest='use_16bit',
        action='store_true',
        help='if true uses 16 bit precision'
    )

    parent_parser.add_argument(
        '--num_gpus',
        type=int,
        default=1
    )

    parent_parser.add_argument(
        '--checkpoint_name',
        type=str,
        default = checkpoint_prefix

    )
    parser = LightningSystem.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    main(hyperparams)