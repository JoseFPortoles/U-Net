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

args = parser.parse_args()

def main(args):
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    wd = args.wd
    input_size = args.input_size
    out_channels = args.num_segment_categories
    weights_path = args.weights_path
    data_root = args.data_root
    output_path = args.output_path
    repartition_set = args.repartition_set
    partition_folder = args.partition_folder
    frozen_encoder = args.frozen_encoder
    lr_scheduler_factor = args.lr_scheduler_factor
    lr_scheduler_patience = args.lr_scheduler_patience
    
    train_loop(num_epochs, batch_size, lr, wd, input_size, out_channels, weights_path, data_root, 
               output_path, repartition_set, partition_folder, frozen_encoder,
               lr_scheduler_factor, lr_scheduler_patience)


if __name__ == '__main__':
    main(args)