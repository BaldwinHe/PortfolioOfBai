import argparse
parser=argparse.ArgumentParser(description="Train/Test Mode")
parser.add_argument('--unity','-u',required=True,help='game path')
parser.add_argument('--actor','-a',default=None,help='path of actor checkpoint')
parser.add_argument('--critic','-c',default=None,help='path of critic checkpoint')
parser.add_argument('--prioritized', action='store_true', default=False,
                    help='whether use Prioritized Experience Replay')
parser.add_argument('--exploration_noise', default=0.1, type=float,help='action noise std')
parser.add_argument('--update_iteration', default=200, type=int)
parser.add_argument('--lr_a3c',default=1e-4,type=float,help='learning rate for A3C')
parser.add_argument('--num-processes', type=int, default=20, metavar='NP',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--seed', type=int, default=1231520, help='default train seed for comparing')
parser.add_argument('--muti', action='store_true', default=False, help='whether use muti agents')
args=parser.parse_args()