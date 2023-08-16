import argparse

def parse_opt():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset name: MNIST, F-MNIST, CIFAR10, HAM')
    
    if parser.parse_args().dataset == 'HAM':
        parser.add_argument('--data_path', type=str, default='./HAM/HAM10000_metadata.csv', help='path of dataset')
        parser.add_argument('--chan', type=int, default=3, help='number of image channel')
        parser.add_argument('--classes', type=int, default=7, help='number of classes')
    elif parser.parse_args().dataset == 'CIFAR10':
        parser.add_argument('--chan', type=int, default=3, help='number of image channel')
        parser.add_argument('--classes', type=int, default=10, help='number of classes')
    elif parser.parse_args().dataset == 'MNIST' or parser.parse_args().dataset == 'F-MNIST':
        parser.add_argument('--chan', type=int, default=1, help='number of image channel')
        parser.add_argument('--classes', type=int, default=10, help='number of classes')

    parser.add_argument('--num_users', type=int, default=9, help='number of clients')
    parser.add_argument('--attacker_id', type=str, default='1', help='id of the malicious client')
    #0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
    parser.add_argument('--attack_before', type=int, default=100, help='attack before this epoch') #0
    parser.add_argument('--attack_after', type=int, default=100, help='attack after this epoch') #=epochs
    parser.add_argument('--bottleneck_num', type=int, default=3, help='number of neurons in bottleneck layer')

    parser.add_argument('--iur', type=float, default=1, help='image update rate') #10 for dataset attack, 10000 for smash attack
    parser.add_argument('--alr', type=float, default=0.01, help='attacking learning rate') #1
    parser.add_argument('--local_epoch', type=int, default=1, help='number of local epoch')

    parser.add_argument('--attack', type=bool, default=True, help='Do adversarial attack')
    parser.add_argument('--dataset_attack', type=bool, default=True, help='attack with fake samples')
    parser.add_argument('--weight_attack', type=bool, default=False, help='attack with fake weights')
    parser.add_argument('--smashed_attack', type=bool, default=False, help='attack with fake smashed data')
    parser.add_argument('--label_attack', type=bool, default=False, help='attack with flipped labels')
    parser.add_argument('--client_agg', type=str, default='avg', help='client aggregation method: avg, trim_mean, median, sparsefed, krum, bulyan')

    parser.add_argument('--epochs', type=int, default=100, help='number of epochs') #200
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate') #1e-4
    parser.add_argument('--iter_num', type=int, default=10, help='number of iteration in the dataset attack')
    parser.add_argument('--L', type=float, default=60, help='norm clipping paprameter in sparsefed') #mnist: >= 20
    parser.add_argument('--k', type=int, default=30000, help='number of coordinates to update each round')
    
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--attack_batch_size', type=int, default=256, help='attack batch size')
    parser.add_argument('--seed', type=int, default=1234, help='number of random seed')
    parser.add_argument('--frac', type=float, default=1.0, help='participation of clients')
    
    args = parser.parse_args()

    return args