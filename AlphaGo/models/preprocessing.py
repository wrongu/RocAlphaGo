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
	planes = np.zeros((state.size, state.size, 3))
	planes[:,:,0] = state.board == state.current_player # own stone
	planes[:,:,1] = state.board == -state.current_player # opponent stone
	planes[:,:,2] = state.board == go.EMPTY # empty space
	return planes

def get_turns_since(state):
	# TODO - this information is not currently stored in GameState?!
	return np.zeros((state.size, state.size, 8))

def get_liberties(state):
	return state.get_liberties()

def get_capture_size(state):
	feature = np.zeros((state.size, state.size, 8))
	# note - left as all zeros if not a legal move
	for a in state.get_legal_moves():
		capture_size_at_a = state.get_capture_size(a)
		# back plane is "7 or more"
		if capture_size_at_a >= 7:
			feature[a[0],a[1],7] = 1
		# otherwise (size 0 through 6) set one-hot vector component
		else:
			feature[a[0],a[1],capture_size_at_a] = 1
	return feature

def get_self_atari_size(state):
	feature = np.zeros((state.size, state.size, 8))
	# note - left as all zeros if not a legal move
	for a in state.get_legal_moves():
		self_atari_size_at_a = state.get_self_atari_size(a)
		# back plane is "7 or more"
		if self_atari_size_at_a >= 7:
			feature[a[0],a[1],7] = 1
		# otherwise (0 through 6) set one-hot vector component
		else:
			feature[a[0],a[1],self_atari_size_at_a] = 1
	return feature

def get_liberties_after(state):
	feature = np.zeros((state.size, state.size, 8))
	# note - left as all zeros if not a legal move
	for a in state.get_legal_moves():
		tmp = state.copy()
		tmp.do_move(a)
		liberties_after_at_a = tmp.get_liberties()[a]
		# back plane is "7 or more"
		if liberties_after_at_a >= 7:
			feature[a[0],a[1],7] = 1
		# otherwise (0 through 6) set one-hot vector component
		else:
			feature[a[0],a[1],liberties_after_at_a] = 1
	return feature

def get_ladder_capture(state):
	feature = np.zeros((state.size, state.size))
	for a in state.get_legal_moves():
		if state.is_ladder_capture(a):
			feature[a] = 1
	return feature

def get_ladder_escape(state):
	feature = np.zeros((state.size, state.size))
	for a in state.get_legal_moves():
		if state.is_ladder_escape(a):
			feature[a] = 1
	return feature

def get_sensibleness(state):
	feature = np.zeros((state.size, state.size))
	moves = state.get_legal_moves()
	# check legality
	for a in moves:
		if state.is_legal(a):
			feature[a] = 1
	# check filling own eye
	for a in moves:
		# multiplication as logical 'and'; a move is sensible if it is legal AND does not fill own eye
		feature[a] *= not(state.fills_eye(a))
	return feature