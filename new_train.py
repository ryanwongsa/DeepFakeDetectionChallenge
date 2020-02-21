from argparse import ArgumentParser, Namespace
from trainer.new_trainer import Trainer

def main(hparams):
    trainer = Trainer(hyperparams)
    trainer.train()
    
if __name__ == '__main__':
    parser = ArgumentParser(parents=[])
    parser.add_argument('--sequence_length', default=1, type=int)
    parser.add_argument('--num_sequences', default=10, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=0, type=int)

    parser.add_argument('--keep_top_k', default=2, type=int)
    parser.add_argument('--threshold_prob', default=0.99, type=float)

    parser.add_argument('--image_size', default=128, type=int)
    parser.add_argument('--margin_factor', default=0.75, type=float)

    parser.add_argument('--num_samples', default=16, type=int)
    parser.add_argument('--is_sequence_classifier', dest='is_sequence_classifier', action='store_true')

    parser.add_argument('--network_name', type=str)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--checkpoint_dir', default=None, type=str)

    parser.add_argument('--grad_acc_num', default=1, type=int)
    parser.add_argument('--lr', type=float, default=0.0003) 

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