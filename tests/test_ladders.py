import unittest
import parseboard
from AlphaGo.go import BLACK, WHITE
from AlphaGo.preprocessing.preprocessing import Preprocess


def is_ladder_capture(state, move):
    pp = Preprocess(["ladder_capture"], size=state.get_size())
    feature = pp.state_to_tensor(state).squeeze()
    return feature[move] == 1


def is_ladder_escape(state, move):
    pp = Preprocess(["ladder_escape"], size=state.get_size())
    feature = pp.state_to_tensor(state).squeeze()
    return feature[move] == 1


class TestLadder(unittest.TestCase):
    """Use interface provided by 'preprocessing' to test ladders
    """

    def test_captured_1(self):
        st, moves = parseboard.parse("d b c . . . .|"
                                     "B W a . . . .|"
                                     ". B . . . . .|"
                                     ". . . . . . .|"
                                     ". . . . . . .|"
                                     ". . . . . W .|")
        st.set_current_player(BLACK)

        # 'a' should catch white in a ladder, but not 'b'
        self.assertTrue(is_ladder_capture(st, moves['a']))
        self.assertFalse(is_ladder_capture(st, moves['b']))

        # 'b' should not be an escape move for white after 'a'
        st.do_move(moves['a'])
        self.assertFalse(is_ladder_escape(st, moves['b']))

        # W at 'b', check 'c' and 'd'
        st.do_move(moves['b'])
        self.assertTrue(is_ladder_capture(st, moves['c']))
        self.assertFalse(is_ladder_capture(st, moves['d']))  # self-atari

    def test_breaker_1(self):
        st, moves = parseboard.parse(". B . . . . .|"
                                     "B W a . . W .|"
                                     "B b . . . . .|"
                                     ". c . . . . .|"
                                     ". . . . . . .|"
                                     ". . . . . W .|"
                                     ". . . . . . .|")
        st.set_current_player(BLACK)

        # 'a' should not be a ladder capture, nor 'b'
        self.assertFalse(is_ladder_capture(st, moves['a']))
        self.assertFalse(is_ladder_capture(st, moves['b']))

        # after 'a', 'b' should be an escape
        st.do_move(moves['a'])
        self.assertTrue(is_ladder_escape(st, moves['b']))

        # after 'b', 'c' should not be a capture
        st.do_move(moves['b'])
        self.assertFalse(is_ladder_capture(st, moves['c']))

    def test_missing_ladder_breaker_1(self):

        st, moves = parseboard.parse(". B . . . . .|"
                                     "B W B . . W .|"
                                     "B a c . . . .|"
                                     ". b . . . . .|"
                                     ". . . . . . .|"
                                     ". W . . . . .|"
                                     ". . . . . . .|")
        st.set_current_player(WHITE)

        # a should not be an escape move for white
        self.assertFalse(is_ladder_escape(st, moves['a']))

        # after 'a', 'b' should still be a capture ...
        st.do_move(moves['a'])
        self.assertTrue(is_ladder_capture(st, moves['b']))
        # ... but 'c' should not
        self.assertFalse(is_ladder_capture(st, moves['c']))

    def test_capture_to_escape_1(self):

        st, moves = parseboard.parse(". O X . . .|"
                                     ". X O X . .|"
                                     ". . O X . .|"
                                     ". . a . . .|"
                                     ". O . . . .|"
                                     ". . . . . .|")
        st.set_current_player(BLACK)

        # 'a' is not a capture because of ataris
        self.assertFalse(is_ladder_capture(st, moves['a']))

    def test_throw_in_1(self):

        st, moves = parseboard.parse("X a O X . .|"
                                     "b O O X . .|"
                                     "O O X X . .|"
                                     "X X . . . .|"
                                     ". . . . . .|"
                                     ". . . O . .|")
        st.set_current_player(BLACK)

        # 'a' or 'b' will capture
        self.assertTrue(is_ladder_capture(st, moves['a']))
        self.assertTrue(is_ladder_capture(st, moves['b']))

        # after 'a', 'b' doesn't help white escape
        st.do_move(moves['a'])
        self.assertFalse(is_ladder_escape(st, moves['b']))

    def test_snapback_1(self):

        st, moves = parseboard.parse(". . . . . . . . .|"
                                     ". . . . . . . . .|"
                                     ". . X X X . . . .|"
                                     ". . O . . . . . .|"
                                     ". . O X . . . . .|"
                                     ". . X O a . . . .|"
                                     ". . X O X . . . .|"
                                     ". . . X . . . . .|"
                                     ". . . . . . . . .|")

        st.set_current_player(WHITE)

        # 'a' is not an escape for white
        self.assertFalse(is_ladder_escape(st, moves['a']))

    def test_two_captures(self):

        st, moves = parseboard.parse(". . . . . .|"
                                     ". . . . . .|"
                                     ". . a b . .|"
                                     ". X O O X .|"
                                     ". . X X . .|"
                                     ". . . . . .|")
        st.set_current_player(BLACK)

        # both 'a' and 'b' should be ladder captures
        self.assertTrue(is_ladder_capture(st, moves['a']))
        self.assertTrue(is_ladder_capture(st, moves['b']))

    def test_two_escapes(self):

        st, moves = parseboard.parse(". . X . . .|"
                                     ". X O a . .|"
                                     ". X c X . .|"
                                     ". O X b . .|"
                                     ". . O . . .|"
                                     ". . . . . .|")

        # place a white stone at c, and reset player to white
        st.do_move(moves['c'], color=WHITE)
        st.set_current_player(WHITE)

        # both 'a' and 'b' should be considered escape moves for white after 'O' at c
        self.assertTrue(is_ladder_escape(st, moves['a']))
        self.assertTrue(is_ladder_escape(st, moves['b']))


if __name__ == '__main__':
    unittest.main()
