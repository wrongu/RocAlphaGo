import os, argparse
import cPickle as pickle
import random
from keras.optimizers import SGD
from AlphaGo.models.policy import CNNPolicy

class supervised_policy_trainer:
    def __init__(self,learning_rate=.003,decay=.0001,batch_size=16,nb_epoch=10):
    	"""Construct a supervised-learning policy trainer.

    	Instance variables:
    	- learning_rate:     Initial learning rate for SGD (default .003)
    	- decay:             Rate of learning rate decay (default .0001)
    	- batch_size:        Number of training samples per SGD minibatch (default 16)
    	- nb_epoch:          Number of iterations through training set (default 10)
        """
        self.learning_rate = learning_rate
        self.decay =  decay
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch

    def train(self,model,train_folder,test_folder,model_folder=None):
        # 1. compile model
        sgd = SGD(lr=self.learning_rate,decay=self.decay)
        model.compile(loss='binary_crossentropy', optimizer=sgd)

        # 2a. construct minibatch getter from train_folder
        train = set()
        file_names = os.listdir(train_folder)
        for file_name in file_names:
            if file_name[-7:] != '.pkl': continue
        else: train.add(file_name)

        batch_generator = self.batch_generator(train,train_folder)

        #2b. load test set from file
        test = () # TODO: construct (num_test_filesXstate,num_test_filesXaction) tuple from test_folder

        # 3. train
        model.fit_generator(generator=batch_generator,samples_per_epoch=len(train),
                            validation_data=test,nb_epoch=self.nb_epoch)

    def batch_generator(self,trainset,train_folder):
        while True:
            sample_filenames = random.sample(trainset,self.batch_size)
            minibatch = () # TODO: construct (self.batch_sizeXstate,self.batch_sizeXaction) tuple from train_folder
            for filename in sample_filenames:
                with open(os.path.join(train_folder,filename),'r') as sample_filename:
                    yield pickle.load(sample_filename) # TODO: change to add to minibatch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform supervised training on a policy network.')
    parser.add_argument("train_folder", help="Path to folder of training samples")
    parser.add_argument("test_folder", help="Path to folder of testing samples")
    parser.add_argument("-model_folder", help="Path to folder where the model params will be saved after each epoch. Default: None",default=None)
    parser.add_argument("-batch_size",help="Number of samples per SGD batch. Default: 16", default=16)
    parser.add_argument("-nb_epoch",help="Total number of iterations on the data. Default: 10",default=10)
    parser.add_argument("-learning_rate",help="How quickly the model learns at first. A (small) number between 0 and 1. Default: .03",default=.03)
    parser.add_argument("-decay",help="The rate at which learning decreases. Default: .0001",default=.0001)
    args = parser.parse_args()

    net = CNNPolicy.create_network(input_dim=32)

    trainer = supervised_policy_trainer(args.learning_rate,args.decay,args.batch_size,args.nb_epoch)
    trainer.train(net,args.train_folder,args.test_folder,args.model_folder)
