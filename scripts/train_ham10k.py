from loops.train_loop_ham10k import train_loop
import argparse


parser = argparse.ArgumentParser(prog='train.py', description='Train U-Net for segmentation task')

parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-3, help='Weight decay.')
parser.add_argument('--input_size', type=int, default=224, help='Lateral size for the (square) input image.')
parser.add_argument('--num_segment_categories', type=int, default=22, help='Number of segmentation categories, including background.')
parser.add_argument('--weights_path', type=str, default=None, help='Weights file path. If =None then initialise net w/ Xavier function.')
parser.add_argument('--data_root', type=str, help='Dataset root folder path.')
parser.add_argument('--output_path', type=str, default='./checkpoints', help='Folder where trained weights are saved.')
parser.add_argument('--repartition_set', action='store_true', help='Repartition dataset.')
parser.add_argument('--partition_folder', type=str, help='Partition folder.')
parser.add_argument('--frozen_encoder', action='store_true', help='Freezes encoder.')
parser.add_argument('--lr_scheduler_factor', type=float, default=0.1, help='lr reduction factor for the lr scheduler.')
parser.add_argument('--lr_scheduler_patience', type=int, default=3, help='lr scheduler patience.')
parser.add_argument('--train_loop', type=str, help='Name of the train loop file without extension.')
parser.add_argument('--num_workers', type=int, help='Num. workers for data loading.')
parser.add_argument('--pin_memory', action='store_true', help='Use pin memory in data loading')

args = parser.parse_args()

def main(args):
    
    train_loop(args.num_epochs, args.batch_size, args.lr, args.wd, args.input_size, args.num_segment_categories, args.weights_path, args.data_root, 
               args.output_path, args.repartition_set, args.partition_folder, args.frozen_encoder,
               args.lr_scheduler_factor, args.lr_scheduler_patience, args.num_workers, args.pin_memory)


if __name__ == '__main__':
    main(args)