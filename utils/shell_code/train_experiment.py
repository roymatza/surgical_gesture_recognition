#Created by Adam Goldbraikh - Scalpel Lab Technion
# parts of the code were adapted from: https://github.com/sj-li/MS-TCN2?utm_source=catalyzex.com
import torch
from Trainer import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import pandas as pd
from datetime import datetime
from termcolor import colored, cprint
import numpy as np
import random

dt_string = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',choices=['APAS'], default="APAS")
parser.add_argument('--task',choices=['gestures', 'multi-task'], default="gestures")
parser.add_argument('--network',choices=['LSTM','GRU'], default="LSTM")
parser.add_argument('--split',choices=['0', '1', '2', '3','4', 'all'], default='all')
parser.add_argument('--features_dim', default='36', type=int)
parser.add_argument('--lr', default=0.00316227766, type=float)
parser.add_argument('--num_epochs', default=40, type=int)
parser.add_argument('--eval_rate', default=1, type=int)
parser.add_argument('--batch_size',default =5, type=int )

parser.add_argument('--dropout', default=0.4, type=float)
parser.add_argument('--num_layers',default =3, type=int )
parser.add_argument('--hidden_dim',default =64, type=int )


parser.add_argument('--normalization', choices=['none'], default='none', type=str)
parser.add_argument('--offline_mode', default=True, type=bool)
parser.add_argument('--project', default=" project name ", type=str)
parser.add_argument('--group', default=dt_string + " group ", type=str)
parser.add_argument('--use_gpu_num',default ="0", type=str )
parser.add_argument('--upload', default=True, type=bool)
parser.add_argument('--debagging', default=True, type=bool)


args = parser.parse_args()
debagging = args.debagging
if debagging:
    args.upload = False
sample_rate = 6  # downsample the frequency to 5Hz
bz = args.batch_size


print(args)
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu_num

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# use the full temporal resolution @ 30Hz




list_of_splits =[]
if len(args.split) == 1:
    list_of_splits.append(int(args.split))

elif args.dataset == "APAS":
    list_of_splits = list(range(0,5))
else:
    raise NotImplemented
dropout = args.dropout
num_layers =args.num_layers
hidden_dim = args.hidden_dim
num_epochs = args.num_epochs
eval_rate = args.eval_rate
features_dim = args.features_dim
lr = args.lr
offline_mode = args.offline_mode
experiment_name = args.group +" task:"  + args.task + " splits: " + args.split +" net: " + args.network + " is Offline: " + str(args.offline_mode)
args.group = experiment_name
# print(colored(experiment_name, "green"))


summaries_dir = "./summaries/" + args.dataset + "/" + experiment_name
if not debagging:
    if not os.path.exists(summaries_dir):
        os.makedirs(summaries_dir)


full_eval_results = pd.DataFrame()
full_train_results = pd.DataFrame()

#root_path = "/datashare/"
root_path = "/home/roym/code/surgical_gesrec/data/datashare/"
for split_num in list_of_splits:
    print("split number: " + str(split_num))
    args.split = str(split_num)

    folds_folder = root_path+args.dataset+"/folds"
    features_path = root_path+args.dataset+"/kinematics_npy/"

    gt_path_gestures = root_path+args.dataset+"/transcriptions_gestures/"
    gt_path_tools_left = root_path+args.dataset+"/transcriptions_tools_left/"
    gt_path_tools_right = root_path+args.dataset+"/transcriptions_tools_right/"

    mapping_gestures_file = root_path+args.dataset+"/mapping_gestures.txt"
    mapping_tool_file = root_path+args.dataset+"/mapping_tools.txt"

    model_dir = "./models/"+args.dataset+"/"+ experiment_name+"/split_"+args.split
    if not debagging:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    file_ptr = open(mapping_gestures_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict_gestures = dict()
    for a in actions:
        actions_dict_gestures[a.split()[1]] = int(a.split()[0])
    num_classes_tools =0
    actions_dict_tools = dict()
    if args.dataset == "APAS":
        file_ptr = open(mapping_tool_file, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for a in actions:
            actions_dict_tools[a.split()[1]] = int(a.split()[0])
        num_classes_tools = len(actions_dict_tools)

    num_classes_gestures = len(actions_dict_gestures)
    if args.task == 'gestures':
        num_classes_list = [num_classes_gestures]
    elif args.task == 'multi-task':
        num_classes_list = [num_classes_gestures, num_classes_tools, num_classes_tools]
        print(num_classes_list)

    trainer = Trainer(features_dim, num_classes_list,hidden_dim=hidden_dim,dropout=dropout,num_layers=num_layers, offline_mode=offline_mode,task=args.task,device=device,network=args.network,debagging=debagging)

    batch_gen = BatchGenerator(num_classes_gestures,num_classes_tools, actions_dict_gestures,actions_dict_tools,features_path,split_num,folds_folder,gt_path_gestures, gt_path_tools_left, gt_path_tools_right, sample_rate=sample_rate,normalization=args.normalization,task=args.task)
    eval_dict ={"features_path":features_path,"actions_dict_gestures": actions_dict_gestures, "actions_dict_tools":actions_dict_tools, "device":device, "sample_rate":sample_rate,"eval_rate":eval_rate,
                "gt_path_gestures":gt_path_gestures, "gt_path_tools_left":gt_path_tools_left, "gt_path_tools_right":gt_path_tools_right,"task":args.task}
    eval_results, train_results = trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr,eval_dict=eval_dict,args=args)


    if not debagging:
        eval_results = pd.DataFrame(eval_results)
        train_results = pd.DataFrame(train_results)
        eval_results = eval_results.add_prefix('split_'+str(split_num)+'_')
        train_results = train_results.add_prefix('split_'+str(split_num)+'_')
        full_eval_results = pd.concat([full_eval_results, eval_results], axis=1)
        full_train_results = pd.concat([full_train_results, train_results], axis=1)
        full_eval_results.to_csv(summaries_dir+"/evaluation_results.csv",index=False)
        full_train_results.to_csv(summaries_dir+"/train_results.csv",index=False)