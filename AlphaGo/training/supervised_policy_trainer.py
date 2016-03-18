import os, argparse
import cPickle as pickle
import random
import numpy as np
from keras.optimizers import SGD
from AlphaGo.models.policy import CNNPolicy

class supervised_policy_trainer:
    def __init__(self,learning_rate=.003,decay=.0001,
                 train_batch_size=16,test_batch_size=None,nb_epoch=10):
    	"""Construct a supervised-learning policy trainer.

    	Instance variables:
    	- learning_rate:     Initial learning rate for SGD (default .003)
    	- decay:             Rate of learning rate decay (default .0001)
    	- train_batch_size:  Number of training samples per SGD minibatch (default 16)
        - test_batch_size:   Number of test samples to use when estimating model accuracy.
                             If None, whole folder is used. (default None)
    	- nb_epoch:          Number of iterations through training set (default 10)
        """
        self.learning_rate = learning_rate
        self.decay =  decay
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.nb_epoch = nb_epoch

    def train(self,model,train_folder,test_folder,model_folder=None):
        # 1. Compile model
        sgd = SGD(lr=self.learning_rate,decay=self.decay)
        model.compile(loss='binary_crossentropy', optimizer=sgd)

        # 2. Construct generators to fetch train and test data.
        X_shape = model.get_config()['layers'][0]['input_shape']
        y_shape = X_shape[-2:] # class labels will always be board x board

        train_size, train_generator = self.setup_generator(train_folder,X_shape,y_shape,
                                            self.train_batch_size,symmetry_transform=True)
        test_size, test_generator = self.setup_generator(test_folder,X_shape,y_shape,self.test_batch_size)

        train_batch_size = train_size if self.train_batch_size == None else self.train_batch_size
        test_batch_size = test_size if self.test_batch_size == None else self.test_batch_size

        # 3. Train
        model.fit_generator(generator=train_generator,samples_per_epoch=train_batch_size,nb_epoch=self.nb_epoch,
                            validation_data=test_generator,nb_val_samples=test_batch_size)

    def setup_generator(self,folder,X_shape,y_shape,num_samples,symmetry_transform=False):
        # Returns number of samples in folder and a generator to access batches of them
        filenames = [filename for filename in os.listdir(folder) if filename[-4:] == '.pkl']
        def generator():
            while True:
                sample_filenames = random.sample(filenames,num_samples)
                batch = self._load_files(folder,sample_filenames,X_shape,y_shape)
                print batch
                # TODO: if symmetry_transform, randomly transform it to some symmetric version of itself
                yield batch
        return(len(filenames),generator())

    def _load_files(self,folder,file_names,X_shape,y_shape):
        X = np.empty((self.train_batch_size,X_shape),dtype='float64')
        y = np.empty((self.train_batch_size,y_shape),dtype='float64')
        for index,filename in enumerate(file_names):
            with open(os.path.join(folder,filename),'r') as sample_filename:
                feature_input, label = pickle.load(sample_filename)
                X[index] = feature_input
                y[index] = label
        return (X,y)

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

    trainer = supervised_policy_trainer(learning_rate=args.learning_rate,decay=args.decay,
                                        train_batch_size=args.batch_size,nb_epoch=args.nb_epoch)
    trainer.train(net,args.train_folder,args.test_folder,args.model_folder)
