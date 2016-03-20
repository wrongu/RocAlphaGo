import os, argparse
import numpy as np
from AlphaGo.preprocessing.preprocessing import Preprocess
from AlphaGo.util import sgf_iter_states
import AlphaGo.go as go
import cPickle as pickle

class game_converter:

	def _encode_label(self, move, size):
		"""Convert move into training label - a board with the given move
		marked as '1' and all other positions '0'
		"""
		one_hot = np.zeros((1, size, size),dtype=bool)
		(x,y) = move
		one_hot[0,x,y] = 1
		return one_hot

	def convert_game(self, file_name, features=None):
		"""Read the given SGF file into an iterable of (input,output) pairs
		for neural networks

		Each input is a GameState converted into one-hot neural net features
		Each output is an action encoded as a categorical choice from positions
		(see _encode_label)
		"""

		with open(file_name,'r') as file_object:
			state_action_iterator = sgf_iter_states(file_object.read())

		if features is None:
			proc = Preprocess()
		else:
			proc = Preprocess(features)

		for (state, move, player) in state_action_iterator:
			if move != go.PASS_MOVE:
				nn_input = proc.state_to_tensor(state)
				nn_output = self._encode_label(move, state.size)
				yield (nn_input, nn_output)
	
	def batch_convert(self, folder_path, features=None):
		"""lazily convert folder of games into training samples
		"""
		file_names = os.listdir(folder_path)
		for file_name in file_names:
			if file_name[-4:] != '.sgf': continue
			print file_name
			training_samples = self.convert_game(os.path.join(folder_path,file_name), features)
			for sample in training_samples:
				yield sample

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Prepare a folder of Go game files for training our neural network model.')
	parser.add_argument("infolder", help="Path to folder containing games")
	parser.add_argument("outfolder", help="Path to target folder.")
	args = parser.parse_args()

	converter = game_converter()
	file_num = 0
	for s_a_tuple in converter.batch_convert(args.infolder,
		features=["board", "ones", "turns_since", "liberties", "capture_size",
		"self_atari_size", "liberties_after","sensibleness", "zeros"]):
		file_name = str(file_num)+".pkl"
		pickle.dump(s_a_tuple, open(os.path.join(args.outfolder,file_name), "wb"))
		file_num += 1
