from AlphaGo.training.supervised_policy_trainer import supervised_policy_trainer
from AlphaGo.models.policy import CNNPolicy
import unittest

class TestSupervisedPolicyTrainer(unittest.TestCase):
    def testTrain(self):
        net = CNNPolicy.create_network(input_dim=32)
        trainer = supervised_policy_trainer()
        trainer.train(net,train_folder='tests/test_data/pickle/train',
                          test_folder = 'tests/test_data/pickle/test')
        #TODO: What should I assert to test training?

if __name__=='__main__':
    unittest.main()
    
