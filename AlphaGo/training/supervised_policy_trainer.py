import os, argparse
import cPickle as pickle
import random
from AlphaGo.models.policy import CNNPolicy

class supervised_trainer:
    def __init__(self,train_folder,model_folder,feature_list):
        self.train_folder = train_folder
        self.model_folder = model_folder
        self.net = CNNPolicy(feature_list)
        self.train_samples = set()
        file_names = os.listdir(train_folder)
        for file_name in file_names:
            if file_name[-7:] != '.pickle': continue
            else: self.train_samples.add(file_name)

    def train(self):
        train_sample_file = random.sample(self.train_samples,1) # randomly select a tensor from the training folder
        with open(train_sample_file,'r') as train_sample:
            pickle.load(train_sample)
            # run forward through model,
            #then backward to compute loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform supervised training on a policy network.')
    parser.add_argument("train_folder", help="Path to folder training samples")
    parser.add_argument("model_folder", help="Path to folder where the model params will be saved each epoch.")
    args = parser.parse_args()

    trainer = supervised_trainer(args.train_folder,args.model_folder,
        ["board", "ones", "turns_since", "liberties", "capture_size",
        "self_atari_size", "liberties_after","sensibleness", "zeros"])
    trainer.train()
