import argparse
parser=argparse.ArgumentParser(description="Train/Test Mode")
parser.add_argument('--unity','-u',required=True,help='game path')
parser.add_argument('--actor','-a',default=None,help='path of actor checkpoint')
parser.add_argument('--critic','-c',default=None,help='path of critic checkpoint')
parser.add_argument('--prioritized', action='store_true', default=False,
                    help='whether use Prioritized Experience Replay')
parser.add_argument('--exploration_noise', default=0.1, type=float,help='action noise std')
parser.add_argument('--update_iteration', default=200, type=int)
args=parser.parse_args()