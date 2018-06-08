import unittest
import parseboard
import numpy as np
import AlphaGo.go as go
from AlphaGo.go import GameState
from AlphaGo.preprocessing.preprocessing import Preprocess


def simple_board():
    gs = GameState(size=7)

    # make a tiny board for the sake of testing and hand-coding expected results
    #
    #         X
    #   0 1 2 3 4 5 6
    #   B W . . . . . 0
    #   B W . . . . . 1
    #   B . . . B . . 2
    # Y . . . B k B . 3
    #   . . . W B W . 4
    #   . . . . W . . 5
    #   . . . . . . . 6
    #
    # where k is a ko position (white was just captured)

    # ladder-looking thing in the top-left
    gs.do_move((0, 0))  # B
    gs.do_move((1, 0))  # W
    gs.do_move((0, 1))  # B
    gs.do_move((1, 1))  # W
    gs.do_move((0, 2))  # B

    # ko position in the middle
    gs.do_move((3, 4))  # W
    gs.do_move((3, 3))  # B
    gs.do_move((4, 5))  # W
    gs.do_move((4, 2))  # B
    gs.do_move((5, 4))  # W
    gs.do_move((5, 3))  # B
    gs.do_move((4, 3))  # W - the ko position
    gs.do_move((4, 4))  # B - does the capture

    return gs


def self_atari_board():
    gs = GameState(size=7)

    # another tiny board for testing self-atari specifically.
    # positions marked with 'a' are self-atari for black
    #
    #         X
    #   0 1 2 3 4 5 6
    #   a W . . . W B 0
    #   . . . . . . . 1
    #   . . . . . . . 2
    # Y . . W . W . . 3
    #   . W B a B W . 4
    #   . . W W W . . 5
    #   . . . . . . . 6
    #
    # current_player = black

    gs.do_move((2, 4), go.BLACK)
    gs.do_move((4, 4), go.BLACK)
    gs.do_move((6, 0), go.BLACK)

    gs.do_move((1, 0), go.WHITE)
    gs.do_move((5, 0), go.WHITE)
    gs.do_move((2, 3), go.WHITE)
    gs.do_move((4, 3), go.WHITE)
    gs.do_move((1, 4), go.WHITE)
    gs.do_move((5, 4), go.WHITE)
    gs.do_move((2, 5), go.WHITE)
    gs.do_move((3, 5), go.WHITE)
    gs.do_move((4, 5), go.WHITE)

    return gs


def capture_board():
    gs = GameState(size=7)

    # another small board, this one with imminent captures
    #
    #         X
    #   0 1 2 3 4 5 6
    #   . . B B . . . 0
    #   . B W W B . . 1
    #   . B W . . . . 2
    # Y . . B . . . . 3
    #   . . . . W B . 4
    #   . . . W . W B 5
    #   . . . . W B . 6
    #
    # current_player = black

    black = [(2, 0), (3, 0), (1, 1), (4, 1), (1, 2), (2, 3), (5, 4), (6, 5), (5, 6)]
    white = [(2, 1), (3, 1), (2, 2), (4, 4), (3, 5), (5, 5), (4, 6)]

    for B in black:
        gs.do_move(B, go.BLACK)
    for W in white:
        gs.do_move(W, go.WHITE)
    gs.set_current_player(go.BLACK)

    return gs


class TestPreprocessingFeatures(unittest.TestCase):
    """Test the functions in preprocessing.py
    """

    def test_get_board(self):
        gs = simple_board()
        pp = Preprocess(["board"], size=7)
        feature = pp.state_to_tensor(gs)[0].transpose((1, 2, 0))

        white_pos = np.asarray([
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]])
        black_pos = np.asarray([
            [1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]])
        empty_pos = np.ones((gs.get_size(), gs.get_size())) - (white_pos + black_pos)

        # check number of planes
        self.assertEqual(feature.shape, (gs.get_size(), gs.get_size(), 3))
        # check return value against hand-coded expectation
        # (given that current_player is white)
        self.assertTrue(np.all(feature == np.dstack((white_pos, black_pos, empty_pos))))

    def test_get_turns_since(self):
        gs = simple_board()
        pp = Preprocess(["turns_since"], size=7)
        feature = pp.state_to_tensor(gs)[0].transpose((1, 2, 0))

        one_hot_turns = np.zeros((gs.get_size(), gs.get_size(), 8))

        rev_moves = list(gs.get_history())
        rev_moves = rev_moves[::-1]

        board = gs.get_board()

        for x in range(gs.get_size()):
            for y in range(gs.get_size()):
                if board[x, y] != go.EMPTY:
                    # find most recent move at x, y
                    age = rev_moves.index((x, y))
                    one_hot_turns[x, y, min(age, 7)] = 1

        self.assertTrue(np.all(feature == one_hot_turns))

    def test_get_liberties(self):
        gs = simple_board()
        pp = Preprocess(["liberties"], size=7)
        feature = pp.state_to_tensor(gs)[0].transpose((1, 2, 0))

        # todo - test liberties when > 8

        one_hot_liberties = np.zeros((gs.get_size(), gs.get_size(), 8))
        # black piece at (4,4) has a single liberty: (4,3)
        one_hot_liberties[4, 4, 0] = 1

        # the black group in the top left corner has 2 liberties
        one_hot_liberties[0, 0:3, 1] = 1
        #     .. as do the white pieces on the left and right of the eye
        one_hot_liberties[3, 4, 1] = 1
        one_hot_liberties[5, 4, 1] = 1

        # the white group in the top left corner has 3 liberties
        one_hot_liberties[1, 0:2, 2] = 1
        #     ...as does the white piece at (4,5)
        one_hot_liberties[4, 5, 2] = 1
        #     ...and the black pieces on the sides of the eye
        one_hot_liberties[3, 3, 2] = 1
        one_hot_liberties[5, 3, 2] = 1

        # the black piece at (4,2) has 4 liberties
        one_hot_liberties[4, 2, 3] = 1

        for i in range(8):
            self.assertTrue(
                np.all(feature[:, :, i] == one_hot_liberties[:, :, i]),
                "bad expectation: stones with %d liberties" % (i + 1))

    def test_get_capture_size(self):
        gs = capture_board()
        pp = Preprocess(["capture_size"], size=7)
        feature = pp.state_to_tensor(gs)[0].transpose((1, 2, 0))

        score_before = gs.get_captures_white()
        one_hot_capture = np.zeros((gs.get_size(), gs.get_size(), 8))

        # There is no capture available; all legal moves are zero-capture
        for (x, y) in gs.get_legal_moves():
            copy = gs.copy()
            copy.do_move((x, y))
            num_captured = copy.get_captures_white() - score_before
            one_hot_capture[x, y, min(7, num_captured)] = 1

        for i in range(8):
            self.assertTrue(
                np.all(feature[:, :, i] == one_hot_capture[:, :, i]),
                "bad expectation: capturing %d stones" % i)

    def test_get_self_atari_size(self):
        gs = self_atari_board()
        pp = Preprocess(["self_atari_size"], size=7)
        feature = pp.state_to_tensor(gs)[0].transpose((1, 2, 0))

        one_hot_self_atari = np.zeros((gs.get_size(), gs.get_size(), 8))
        # self atari of size 1 at position 0,0
        one_hot_self_atari[0, 0, 0] = 1
        # self atari of size 3 at position 3,4
        one_hot_self_atari[3, 4, 2] = 1

        self.assertTrue(np.all(feature == one_hot_self_atari))

    def test_get_self_atari_size_cap(self):
        gs = capture_board()
        pp = Preprocess(["self_atari_size"], size=7)
        feature = pp.state_to_tensor(gs)[0].transpose((1, 2, 0))

        one_hot_self_atari = np.zeros((gs.get_size(), gs.get_size(), 8))
        # self atari of size 1 at the ko position and just below it
        one_hot_self_atari[4, 5, 0] = 1
        one_hot_self_atari[3, 6, 0] = 1
        # self atari of size 3 at bottom corner
        one_hot_self_atari[6, 6, 2] = 1

        self.assertTrue(np.all(feature == one_hot_self_atari))

    def test_get_liberties_after(self):
        gs = simple_board()
        pp = Preprocess(["liberties_after"], size=7)
        feature = pp.state_to_tensor(gs)[0].transpose((1, 2, 0))

        one_hot_liberties = np.zeros((gs.get_size(), gs.get_size(), 8))

        # TODO (?) hand-code?
        for (x, y) in gs.get_legal_moves():
            copy = gs.copy()
            copy.do_move((x, y))

            liberty = copy.get_liberty()

            libs = liberty[x, y]
            if libs < 7:
                one_hot_liberties[x, y, libs - 1] = 1
            else:
                one_hot_liberties[x, y, 7] = 1

        for i in range(8):
            self.assertTrue(
                np.all(feature[:, :, i] == one_hot_liberties[:, :, i]),
                "bad expectation: stones with %d liberties after move" % (i + 1))

    def test_get_liberties_after_cap(self):
        """
           A copy of test_get_liberties_after but where captures are imminent
        """

        gs = capture_board()
        pp = Preprocess(["liberties_after"], size=7)
        feature = pp.state_to_tensor(gs)[0].transpose((1, 2, 0))

        one_hot_liberties = np.zeros((gs.get_size(), gs.get_size(), 8))

        for (x, y) in gs.get_legal_moves():
            copy = gs.copy()
            copy.do_move((x, y))

            liberty = copy.get_liberty()

            libs = liberty[x, y]
            one_hot_liberties[x, y, min(libs - 1, 7)] = 1

        for i in range(8):
            self.assertTrue(
                np.all(feature[:, :, i] == one_hot_liberties[:, :, i]),
                "bad expectation: stones with %d liberties after move" % (i + 1))

    def test_get_ladder_capture(self):
        gs, moves = parseboard.parse(". . . . . . .|"
                                     "B W a . . . .|"
                                     ". B . . . . .|"
                                     ". . . . . . .|"
                                     ". . . . . . .|"
                                     ". . . . . W .|")
        pp = Preprocess(["ladder_capture"], size=7)
        feature = pp.state_to_tensor(gs)[0, 0]  # 1D tensor; no need to transpose

        expectation = np.zeros((gs.get_size(), gs.get_size()))
        expectation[moves['a']] = 1

        self.assertTrue(np.all(expectation == feature))

    def test_get_ladder_escape(self):
        # On this board, playing at 'a' is ladder escape because there is a breaker on the right.
        gs, moves = parseboard.parse(". B B . . . .|"
                                     "B W a . . . .|"
                                     ". B . . . . .|"
                                     ". . . . . W .|"
                                     ". . . . . . .|"
                                     ". . . . . . .|")
        pp = Preprocess(["ladder_escape"], size=7)
        gs.set_current_player(go.WHITE)
        feature = pp.state_to_tensor(gs)[0, 0]  # 1D tensor; no need to transpose

        expectation = np.zeros((gs.get_size(), gs.get_size()))
        expectation[moves['a']] = 1

        self.assertTrue(np.all(expectation == feature))

    def test_two_escapes(self):
        gs, moves = parseboard.parse(". . X . . .|"
                                     ". X O a . .|"
                                     ". X c X . .|"
                                     ". O X b . .|"
                                     ". . O . . .|"
                                     ". . . . . .|")

        # place a white stone at c, and reset player to white
        gs.do_move(moves['c'], color=go.WHITE)
        gs.set_current_player(go.WHITE)

        pp = Preprocess(["ladder_escape"], size=6)
        gs.set_current_player(go.WHITE)
        feature = pp.state_to_tensor(gs)[0, 0]  # 1D tensor; no need to transpose

        # both 'a' and 'b' should be considered escape moves for white after 'O' at c

        expectation = np.zeros((gs.get_size(), gs.get_size()))
        expectation[moves['a']] = 1
        expectation[moves['b']] = 1

        self.assertTrue(np.all(expectation == feature))

    def test_get_sensibleness(self):
        gs, moves = parseboard.parse("x B . . W . . . .|"
                                     "B B W . . W . . .|"
                                     ". W B B W W . . .|"
                                     ". B y B W W . . .|"
                                     ". B B z B W . . .|"
                                     ". . B B B W . . .|"
                                     ". . . . . . . . W|"
                                     ". . . . . . . . W|"
                                     ". . . . . . . W s|")
        gs.set_current_player(go.BLACK)

        pp = Preprocess(["sensibleness"], size=9)
        feature = pp.state_to_tensor(gs)[0, 0]  # 1D tensor; no need to transpose

        expectation = np.zeros((gs.get_size(), gs.get_size()), dtype=int)

        for (x, y) in gs.get_legal_moves():
            expectation[x, y] = 1

        # 'x', 'y', and 'z' are eyes - remove them from 'sensible' moves
        expectation[moves['x']] = 0
        expectation[moves['y']] = 0
        expectation[moves['z']] = 0

        # 's' is suicide - should not be legal
        expectation[moves['s']] = 0

        self.assertTrue(np.all(expectation == feature))

    def test_get_legal(self):
        gs = simple_board()
        pp = Preprocess(["legal"], size=7)
        feature = pp.state_to_tensor(gs)[0, 0]  # 1D tensor; no need to transpose

        expectation = np.zeros((gs.get_size(), gs.get_size()))
        for (x, y) in gs.get_legal_moves():
            expectation[x, y] = 1
        self.assertTrue(np.all(expectation == feature))

    def test_feature_concatenation(self):
        gs = simple_board()
        pp = Preprocess(["board", "sensibleness", "capture_size"], size=7)
        feature = pp.state_to_tensor(gs)[0].transpose((1, 2, 0))

        expectation = np.zeros((gs.get_size(), gs.get_size(), 3 + 1 + 8))

        board = gs.get_board()

        # first three planes: board
        expectation[:, :, 0] = (board == go.WHITE) * 1
        expectation[:, :, 1] = (board == go.BLACK) * 1
        expectation[:, :, 2] = (board == go.EMPTY) * 1

        # 4th plane: sensibleness (as in test_get_sensibleness)
        for (x, y) in gs.get_legal_moves():
            if not (gs.is_eye((x, y), go.WHITE)):
                expectation[x, y, 3] = 1

        # 5th through 12th plane: capture size (all zero-capture)
        for (x, y) in gs.get_legal_moves():
            expectation[x, y, 4] = 1

        self.assertTrue(np.all(expectation == feature))


if __name__ == '__main__':
    unittest.main()
