import os, argparse
import numpy as np
from AlphaGo.models.preprocessing import Preprocess
import AlphaGo.go as go
from sgflib.sgflib import SGFParser, GameTreeEndError

class game_converter:
    def __init__(self):
        self.index_at = {'a':0,'b':1,'c':2,'d':3,
                         'e':4,'f':5,'g':6,'h':7,
                         'i':8,'j':9,'k':10,'l':11,
                         'm':12,'n':13,'o':14,'p':15,
                         'q':16,'r':17,'s':18}

    # convert move into board indices
    def parse_raw_move(self,raw_move):
        pos = list(str(raw_move)[3:5])
        col = self.index_at[pos[0]]
        row = self.index_at[pos[1]]
        return (row,col)

    # convert indices into 19x19 training label
    def encode_label(self,move):
        # convert move to one-hot encoding
        one_hot = np.zeros((19,19),dtype=bool)
        one_hot[move[1]][move[0]] = 1
        return one_hot

    # convert full game into training samples
    def convert_game(self,file_name):
        with open(file_name,'r') as file_object:
            sgf_object = SGFParser(file_object.read())
        c = sgf_object.parse().cursor()
        tensors = []
        actions = []
        gs = go.GameState()
        proc = Preprocess()
        while True:
            try:
                move = self.parse_raw_move(c.next())
                actions.append(self.encode_label(move))
                gs.do_move(move)
                tensors.append(proc.state_to_tensor(gs))
            except GameTreeEndError:
                # remove last board state since it has no label
                states = states[0:-1]
                break
        return zip(tensors, actions)

    # lazily convert folder of games into training samples
    def batch_convert(self,folder_path):
        file_names = os.listdir(folder_path)
        for file_name in file_names:
            if file_name == '.DS_Store': continue # OSX compatibility
            print file_name
            training_samples = self.convert_game(os.path.join(folder_path,file_name))
            for sample in training_samples:
                yield sample

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare a folder of Go game files for training our neural network model.')
    parser.add_argument("infolder", help="Relative path to folder containing games")
    parser.add_argument("outfolder", help="Relative path to target folder. Will be created if it does not exist.")
    args = parser.parse_args()

    converter = game_converter()
    for state,action in converter.batch_convert(args.infolder):
        pass # write to file
