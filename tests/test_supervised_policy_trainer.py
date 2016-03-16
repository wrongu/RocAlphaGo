from AlphaGo.training.supervised_policy_trainer import supervised_policy_trainer
from AlphaGo.models.policy import CNNPolicy
import os

import numpy as np
import unittest

class TestSupervisedPolicyTrainer(unittest.TestCase):
    def testTrain(self):
        net = CNNPolicy.create_network(input_dim=32)
        trainer = supervised_policy_trainer(samples_per_epoch=1,nb_epoch=1)
        trainer.train(net,model_folder='tests/test_data',train_folder='tests/test_data/pickle')
        #TODO: What should I assert to test training?

if __name__=='__main__':
    unittest.main()
