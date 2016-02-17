import numpy as np
import AlphaGo.go as go

# named features and their sizes are defined here
FEATURES = {
	"board" : {
		"size": 3,
		"function": get_board
	},
	"ones" : {
		"size": 1,
		"function": lambda ones: np.zeros((state.size, state.size))
	},
	"turns_since" : {
		"size": 8,
		"function": get_turns_since
	},
	"liberties" : {
		"size": 8,
		"function": get_liberties
	},
	"capture_size", : {
		"size": 8,
		"function": get_capture_size
	},
	"self_atari_size" : {
		"size": 8,
		"function": get_self_atari_size
	},
	"liberties_after" : {
		"size": 8,
		"function": get_liberties_after
	},
	"ladder_capture" : {
		"size": 1,
		"function": get_ladder_capture
	},
	"ladder_escape", : {
		"size": 1,
		"function": get_ladder_escape
	},
	"sensibleness" : {
		"size": 1,
		"function": get_sensibleness
	},
	"zeros" : {
		"size": 1,
		"function": lambda state: np.zeros((state.size, state.size))
	}
}

DEFAULT_FEATURES = ["board", "ones", "turns_since", "liberties", "capture_size",
	"self_atari", "liberties_after", "ladder_capture", "ladder_escape",
	"sensibleness", "zeros"]

class Preprocess(object):
	"""a class to convert from AlphaGo GameState objects to tensors of one-hot
	features for NN inputs
	"""

	def __init__(self, feature_list=DEFAULT_FEATURES):
		"""create a preprocessor object that will concatenate together the 
		given list of features
		"""

		self.output_dim = 0
		self.processors = [None] * len(feature_list)
		for i in range(len(feature_list)):
			feat = feature_list[i].lower()
			if feat in FEATURES:
				self.processors[i] = FEATURES[i]["function"]
				self.output_dim += FEATURES[i]["size"]

	def state_to_tensor(self, state):
		feat_tensors = (proc(state) for proc in self.processors)
		# concatenate along the depth dimension
		return np.concatenate(feat_tensors, axis=2)

##
## individual feature functions (state --> tensor) begin here
##

def get_board(state):
	"""A feature encoding WHITE BLACK and EMPTY on separate planes, but plane 0
	always refers to the current player and plane 1 to the opponent
	"""
	planes = np.zeros((state.size, state.size, 3))
	planes[:,:,0] = state.board == state.current_player # own stone
	planes[:,:,1] = state.board == -state.current_player # opponent stone
	planes[:,:,2] = state.board == go.EMPTY # empty space
	return planes

def get_turns_since(state, maximum=8):
	"""A feature encoding the age of the stone at each location up to 'maximum'

	Note: 
	- the [maximum-1] plane is used for any stone with age greater than or equal to maximum
	- EMPTY locations are all-zero features
	"""
	planes = np.zeros((state.size, state.size, maximum))
	depth = 0
	# loop backwards over history and place a 1 in plane 0
	# for the most recent move, a 1 in plane 1 for two moves ago, etc..
	for move in state.history[::-1]:
		if move is not go.PASS_MOVE:
			(x,y) = move
			planes[x,y,depth] = 1
		# increment depth if there are more planes available
		# (the last plane serves as the "maximum-1 or more" feature)
		if depth < maximum-1:
			depth += 1
	return planes

def get_liberties(state, maximum=8):
	"""A feature encoding the number of liberties of the group connected to the stone at
	each location

	Note: 
	- there is no zero-liberties plane; the 0th plane indicates groups in atari
	- the [maximum-1] plane is used for any stone with liberties greater than or equal to maximum
	- EMPTY locations are all-zero features
	"""
	planes = np.zeros((state.size, state.size, maximum))
	liberties = state.update_current_liberties()
	for i in range(maximum):
		# single liberties in plane zero (groups won't have zero), double liberties in plane one, etc
		planes[liberties == i+1, i] = 1
	# the "maximum-or-more" case on the backmost plane
	planes[liberties >= maximum, maximum-1] = 1
	return planes

def get_capture_size(state, maximum=8):
	"""A feature encoding the number of opponent stones that would be captured by planing at each location,
	up to 'maximum'

	Note: 
	- the [maximum-1] plane is used for any capturable group of size greater than or equal to maximum-1
	- the 0th plane is used for legal moves that would not result in capture
	- illegal move locations are all-zero features
	"""
	planes = np.zeros((state.size, state.size, maximum))
	# check difference in size after doing each move
	for (x,y) in state.get_legal_moves():
		copy = state.copy()
		copy.do_move(a)
		if state.current_player == go.BLACK:
			n_captured = copy.num_white_prisoners - state.num_white_prisoners
		else:
			n_captured = copy.num_black_prisoners - state.num_black_prisoners
		planes[x,y,min(n_captured,maximum-1)] = 1
	return planes

def get_self_atari_size(state, maximum=8):
	# TODO - not sure if this refers to size of atari'd group after doing this move
	# (and whether it's undefined if playing there would get the group out of atari)
	# or size of capture if opponent were to play here
	raise NotImplementedError()

def get_liberties_after(state, maximum=8):
	"""A feature encoding what the number of liberties *would be* of the group connected to
	the stone *if* played at a location

	Note: 
	- there is no zero-liberties plane; the 0th plane indicates groups in atari
	- the [maximum-1] plane is used for any stone with liberties greater than or equal to maximum
	- illegal move locations are all-zero features
	"""
	feature = np.zeros((state.size, state.size, maximum))
	# note - left as all zeros if not a legal move
	for (x,y) in state.get_legal_moves():
		tmp = state.copy()
		tmp.do_move((x,y))
		liberties_after_at_a = tmp.update_current_liberties()[x,y]
		feature[x,y,min(maximum-1,liberties_after_at_a)] = 1
	return feature

def get_ladder_capture(state):
	raise NotImplementedError()

def get_ladder_escape(state):
	raise NotImplementedError()

def get_sensibleness(state):
	"""A move is 'sensible' if it is legal and if it does not fill the current_player's own eye
	"""
	feature = np.zeros((state.size, state.size))
	moves = state.get_legal_moves()
	for a in moves:
		if state.is_legal(a) and not state.is_eye(a, state.current_player):
			feature[a] = 1
	return feature