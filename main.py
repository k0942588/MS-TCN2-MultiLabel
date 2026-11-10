#!/usr/bin/python2.7

import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random
import re

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="gtea")
parser.add_argument('--split', default='1')

parser.add_argument('--features_dim', default='2048', type=int)
parser.add_argument('--bz', default='1', type=int)
parser.add_argument('--lr', default='0.0005', type=float)

parser.add_argument('--num_f_maps', default='64', type=int)

# Need input
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--num_layers_PG', default='11', type=int)
parser.add_argument('--num_layers_R', default='10', type=int)
parser.add_argument('--num_R', default='3', type=int)

# david
parser.add_argument('--bce_pos_weight', type=float, default=1, help='')
parser.add_argument('--smoothing_weight', type=float, default=0.15)
parser.add_argument('--smoothing_tau', type=float, default=4.0)
parser.add_argument('--refinement_input', type=str, default='sigmoid')
parser.add_argument('--save_best_only', action='store_true')
parser.add_argument('--save_every', type=int, default=1)
parser.add_argument('--save_optimizer', action='store_true')
parser.add_argument('--use_best_model', action='store_true')
parser.add_argument('--predict_model_path', type=str, default=None)

# Choose splits and output folders from train.sh / predict.sh.
parser.add_argument('--split_dir', type=str, default=None)
parser.add_argument('--train_set', type=str, default='train')
parser.add_argument('--val_set', type=str, default='val')
parser.add_argument('--test_set', type=str, default='test')
parser.add_argument('--predict_set', type=str, default='val')
parser.add_argument('--model_dir', type=str, default=None)
parser.add_argument('--results_dir', type=str, default=None)

parser.add_argument(
    '--stage_diagnostics',
    action='store_true',
    help='Output stage-wise PG/R mAP and per-class AP for diagnosis.'
)
#

args = parser.parse_args()

# Checkpoints may have a different refinement architecture than main.py defaults.
# Example: best_R1_logits_pw1_s0.15_tau4_lr0.0003_bz1.model
if args.action == 'predict' and args.predict_model_path:
    checkpoint_name = os.path.basename(args.predict_model_path)
    match = re.search(r'(?:^|_)R(\d+)_(sigmoid|logits|PGonly)(?:_|\.|$)', checkpoint_name)
    if match:
        args.num_R = int(match.group(1))
        args.refinement_input = (
            'sigmoid' if match.group(2) == 'PGonly' else match.group(2)
        )
        print(f"Checkpoint architecture: num_R={args.num_R}, "
              f"refinement_input={args.refinement_input}")
    else:
        print('WARNING: Cannot infer architecture from checkpoint filename; '
              'using --num_R and --refinement_input')

num_epochs = args.num_epochs
features_dim = args.features_dim
bz = args.bz
lr = args.lr

num_layers_PG = args.num_layers_PG
num_layers_R = args.num_layers_R
num_R = args.num_R
num_f_maps = args.num_f_maps

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2
elif args.dataset == "carom":
    sample_rate = 1

split_dir = args.split_dir or "./data/"+args.dataset+"/splits"
vid_list_file = os.path.join(split_dir, args.train_set+".split"+args.split+".bundle")
vid_list_file_tst = os.path.join(split_dir, args.test_set+".split"+args.split+".bundle")
# david
vid_list_file_val = os.path.join(split_dir, args.val_set+".split"+args.split+".bundle")
vid_list_file_predict = os.path.join(split_dir, args.predict_set+".split"+args.split+".bundle")
#
features_path = "./data/"+args.dataset+"/features/"
gt_path = "./data/"+args.dataset+"/groundTruth/"

mapping_file = "./data/"+args.dataset+"/mapping.txt"

model_dir = args.model_dir or "./models/"+args.dataset+"/split_"+args.split
results_dir = args.results_dir or "./results/"+args.dataset+"/split_"+args.split

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

with open(mapping_file, 'r') as f:
    actions = [
        line.strip() for line in f if line.strip()
    ]

actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

num_classes = len(actions_dict)
trainer = Trainer(args, num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes, args.dataset, args.split, device=device)
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)
    print(f"Train videos: {len(batch_gen.list_of_examples)}")

    # david
    # validate
    batch_gen_val = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen_val.read_data(vid_list_file_val)
    print(f"Val videos: {len(batch_gen_val.list_of_examples)}")
    #

    trainer.train(model_dir, batch_gen, batch_gen_val, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

# if args.action == "predict":
#     #trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate)
#     trainer.predict(
#         model_dir, results_dir, features_path, vid_list_file_tst, num_epochs,
#         actions_dict, device, sample_rate, gt_path, mapping_file,
#         use_best_model=args.use_best_model
#     )
if args.action == "predict":
    trainer.predict(
        model_dir, results_dir, features_path, vid_list_file_predict,
        num_epochs, actions_dict, device, sample_rate,
        gt_path, mapping_file,
        use_best_model=args.use_best_model,
        stage_diagnostics=args.stage_diagnostics,
        model_path=args.predict_model_path
    )
