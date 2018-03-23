import argparse

parser = argparse.ArgumentParser(description='Pytorch Scene Recognition Training')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                   help='input batch size for training(default:100)')
parser.add_argument('--num_classes', type=int, default=80, metavar='N',
                    help='the total classes kinds number of scene classifier')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='N',
                   help='input batch size for testing')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                   help='number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', type=str, default='', metavar='PATH',
                    help='path to latest checkpoint (default:none)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                   help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                   help='SGD momentum')
parser.add_argument('--weight_decay', '--wd', type=float, default=1e-4, metavar='W',
                   help='wight decay(default: 1e-4)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                   help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                   help='random seed(default:1)')
parser.add_argument('--log_inteval', type=int, default=50, metavar='N',
                   help='how many batches to wait before logging training status')
parser.add_argument('--arch', '-a', default='alexnet', metavar='ARCH',
                   help='the deep model architecture (default: alexnet)')

parser.add_argument('--train_log', type=str, default='/home/haoyanlong/AI/logs/alexnet_log/train',
                    help='the train parameters logging')
parser.add_argument('--val_log', type=str, default='/home/haoyanlong/AI/logs/alexnet_log/val',
                    help='the validation curracy logging')
parser.add_argument('--checkpoint_path', type=str, default='/home/haoyanlong/AI/modelcheckpoint/alexnet/',
                    help='the directory of model checkpoint saved')

def main():

    global args, best_prec1
    args = parser.parse_args()
    print args

if __name__ == '__main__':
    main()