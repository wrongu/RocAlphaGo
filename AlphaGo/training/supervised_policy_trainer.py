import os, argparse
import cPickle as pickle
import random
from AlphaGo.models.policy import CNNPolicy

class supervised_policy_trainer:
    def __init__(self,samples_per_epoch,nb_epoch):
        self.samples_per_epoch=samples_per_epoch
        self.nb_epoch=nb_epoch

    def train(self,model,model_folder,train_folder):
        # 1. compile model
        model.compile(loss='binary_crossentropy', optimizer='sgd')

        # 2. get training data
        train_samples = set()
        file_names = os.listdir(train_folder)
        for file_name in file_names:
            if file_name[-7:] != '.pickle': continue
            else: train_samples.add(file_name)

        # 3. train
        sample_generator = self.sample_generator(train_folder,train_samples)
        model.fit_generator(sample_generator,self.samples_per_epoch,self.nb_epoch)

    def sample_generator(self,train_folder,train_samples):
        while True:
            sample_filename = random.sample(train_samples,1)[0]
            with open(os.path.join(train_folder,sample_filename),'r') as sample_filename:
                yield pickle.load(sample_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform supervised training on a policy network.')
    parser.add_argument("train_folder", help="Path to folder training samples")
    parser.add_argument("model_folder", help="Path to folder where the model params will be saved each epoch.")
    parser.add_argument("-samples_per_epoch",help="Number of samples to process before starting a new epoch.",default=16)
    parser.add_argument("-nb_epoch",help="Total number of iterations on the data.",default=100)
    args = parser.parse_args()

    net = CNNPolicy.create_network() # needs kwargs
    trainer = supervised_policy_trainer(args.samples_per_epoch,args.nb_epoch)
    trainer.train(net,args.model_folder,args.train_folder)
