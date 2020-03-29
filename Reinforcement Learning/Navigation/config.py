import argparse
parser=argparse.ArgumentParser(description="Train/Test Mode")
parser.add_argument('--unity','-u',required=True,help='game path')
parser.add_argument('--checkpoint','-c',default=None,help='path of checkpoint')
parser.add_argument('--double', action='store_true', default=False, help='whether use double DQN')
args=parser.parse_args()