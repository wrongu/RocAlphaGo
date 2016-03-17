import os, argparse
import cPickle as pickle
import random
from keras.optimizers import SGD
from AlphaGo.models.policy import CNNPolicy

class supervised_policy_trainer:
    
	"""Construct a supervised-learning policy trainer.

	Instance variables:
	- learning_rate:     Initial learning rate for SGD (default .003)
	- decay:             Rate of learning rate decay (default .0001)
	- batch_size:        Number of training samples per SGD minibatch (default 16)
	- nb_epoch:          Number of iterations through training set (default 10)"""

    def __init__(self,learning_rate=.003,decay=.0001,batch_size=16,nb_epoch=10):
		self.learning_rate = learning_rate
        self.decay =  decay
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch

    def train(self,model,train_folder,test_folder,model_folder=None):
        # 1. compile model
        sgd = SGD(lr=self.params['learning_rate'],decay=self.params['decay'])
        model.compile(loss='binary_crossentropy', optimizer=sgd)

        # 2. construct sample generator
        batch_generator = self.batch_generator(model_folder,train_folder,test_folder,
                                               batch_size=self.params['batch_size'])
        # 3. train
        model.fit_generator(generator=self.batch_generator,
            samples_per_epoch=len(trainset),nb_epoch=self.nb_epoch)

    def batch_generator(self,model_folder,train_folder,test_folder,batch_size):
        # ONLY SERIALIZE IF MODEL FOLDER IS NOT NONE
        trainset = set()
        file_names = os.listdir(train_folder)
        for file_name in file_names:
            if file_name[-7:] != '.pkl': continue
        else: trainset.add(file_name)
        samples_per_epoch = len(trainset)
        while True:
            sample_filename = random.sample(trainset,1)[0]
            with open(os.path.join(train_folder,sample_filename),'r') as sample_filename:
                yield pickle.load(sample_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform supervised training on a policy network.')
    parser.add_argument("train_folder", help="Path to folder of training samples")
    parser.add_argument("test_folder", help="Path to folder testing samples")
    parser.add_argument("-model_folder", help="Path to folder where the model params will be saved after each epoch.",default=None)
    parser.add_argument("-batch_size",help="Number of samples per SGD batch.", default=16)
    parser.add_argument("-nb_epoch",help="Total number of iterations on the data.",default=10)
    args = parser.parse_args()

    net = CNNPolicy.create_network(input_dim=32)

    # trainer = supervised_policy_trainer(APPROPRIATE ARGS)
    # trainer.train(net,args.train_folder,args.test_folder,args.model_folder)
