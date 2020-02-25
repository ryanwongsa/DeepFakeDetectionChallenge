from argparse import ArgumentParser, Namespace
from trainer.audio_trainer import AudioTrainer

def main(hparams):
    trainer = AudioTrainer(hyperparams)
    trainer.train()
    
if __name__ == '__main__':
    parser = ArgumentParser(parents=[])
    parser.add_argument('--mixup', dest='mixup', action='store_true')
    parser.add_argument('--cutmix', dest='cutmix', action='store_true')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pos_weight_factor', type=float, default=0.5) 

    parser.add_argument('--network_name', type=str)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--checkpoint_dir', default=None, type=str)

    parser.add_argument('--grad_acc_num', default=1, type=int)
    parser.add_argument('--lr', type=float, default=0.0003) 
    parser.add_argument('--pos_weight_factor', type=float, default=1.0) 
    
    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--train_meta_file', type=str)
    parser.add_argument('--valid_dir', type=str)
    parser.add_argument('--valid_meta_file', type=str)

    parser.add_argument('--project_name', default=None, type=str)
    parser.add_argument('--run_name', default=None, type=str)

    parser.add_argument('--optimizer_name', type=str, default='default')
    parser.add_argument('--scheduler_name', type=str, default='default')
    parser.add_argument('--criterion_name', type=str, default='default')
    parser.add_argument('--tuning_type', type=str, default='default')
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--use_amp', dest='use_amp', action='store_true')
    parser.add_argument('--load_model_only', dest='load_model_only', action='store_true')

    hyperparams = parser.parse_args()
    
    main(hyperparams)