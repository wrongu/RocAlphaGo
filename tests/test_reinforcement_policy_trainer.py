import os
from AlphaGo.training.reinforcement_policy_trainer import run_training
import unittest


class TestReinforcementPolicyTrainer(unittest.TestCase):
	def testTrain(self):
		model = 'tests/test_data/minimodel.json'
		init_weights = 'tests/test_data/hdf5/random_minimodel_weights.hdf5'
		output = 'tests/test_data/.tmp.rl.training/'
		args = [model, init_weights, output, '--game-batch', '1', '--iterations', '1']
		run_training(args)

		os.remove(os.path.join(output, 'metadata.json'))
		os.remove(os.path.join(output, 'weights.00000.hdf5'))
		os.remove(os.path.join(output, 'weights.00001.hdf5'))
		os.rmdir(output)

if __name__ == '__main__':
	unittest.main()
