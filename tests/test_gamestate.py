import parseboard
import unittest
import AlphaGo.go as go
from AlphaGo.go import GameState
from AlphaGo.util import flatten_idx


class TestKo(unittest.TestCase):

    def test_standard_ko(self):

        gs = GameState(size=9)

        gs.do_move((1, 0))  # B
        gs.do_move((2, 0))  # W
        gs.do_move((0, 1))  # B
        gs.do_move((3, 1))  # W
        gs.do_move((1, 2))  # B
        gs.do_move((2, 2))  # W
        gs.do_move((2, 1))  # B

        gs.do_move((1, 1))  # W trigger capture and ko

        self.assertEqual(gs.get_captures_black(), 1)
        self.assertEqual(gs.get_captures_white(), 0)

        self.assertFalse(gs.is_legal((2, 1)))

        gs.do_move((5, 5))
        gs.do_move((5, 6))

        self.assertTrue(gs.is_legal((2, 1)))

    def test_snapback_is_not_ko(self):

        gs = GameState(size=5)

        # B o W B .
        # W W B . .
        # . . . . .
        # . . . . .
        # . . . . .
        # here, imagine black plays at 'o' capturing
        # the white stone at (2, 0). White may play
        # again at (2, 0) to capture the black stones
        # at (0, 0), (1, 0). this is 'snapback' not 'ko'
        # since it doesn't return the game to a
        # previous position
        B = [(0, 0), (2, 1), (3, 0)]
        W = [(0, 1), (1, 1), (2, 0)]
        for (b, w) in zip(B, W):
            gs.do_move(b)
            gs.do_move(w)
        # do the capture of the single white stone
        gs.do_move((1, 0))
        # there should be no ko
        self.assertIsNone(gs.get_ko_location())
        self.assertTrue(gs.is_legal((2, 0)))
        # now play the snapback
        gs.do_move((2, 0))
        # check that the numbers worked out
        self.assertEqual(gs.get_captures_black(), 2)
        self.assertEqual(gs.get_captures_white(), 1)

    def test_positional_superko(self):

        # test with enforce_superko=False
        gs = GameState(size=9, enforce_superko=False)

        move_list = [(0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4), (2, 2), (3, 4), (2, 1), (3, 3),
                     (3, 1), (3, 2), (3, 0), (4, 2), (1, 1), (4, 1), (8, 0), (4, 0), (8, 1), (0, 2),
                     (8, 2), (0, 1), (8, 3), (1, 0), (8, 4), (2, 0), (0, 0)]

        for move in move_list:
            gs.do_move(move)
        self.assertTrue(gs.is_legal((1, 0)))

        # test with enforce_superko=True
        gs = GameState(size=9, enforce_superko=True)
        for move in move_list:
            gs.do_move(move)
        self.assertFalse(gs.is_legal((1, 0)))


class TestEye(unittest.TestCase):

    def test_true_eye(self):

        gs = GameState(size=7)

        gs.do_move((1, 0), go.BLACK)
        gs.do_move((0, 1), go.BLACK)

        # false eye at 0, 0
        self.assertFalse(gs.is_eye((0, 0), go.BLACK))

        # make it a true eye by turning the corner (1, 1) into an eye itself
        gs.do_move((1, 2), go.BLACK)
        gs.do_move((2, 1), go.BLACK)
        gs.do_move((2, 2), go.BLACK)
        gs.do_move((0, 2), go.BLACK)

        # is eyeish function does not exist
        self.assertTrue(gs.is_eye((0, 0), go.BLACK))
        self.assertTrue(gs.is_eye((1, 1), go.BLACK))

    def test_eye_recursion(self):
        # a checkerboard pattern of black is 'technically' all true eyes
        # mutually supporting each other

        gs = GameState(size=7)

        for x in range(gs.get_size()):
            for y in range(gs.get_size()):
                if (x + y) % 2 == 1:
                    gs.do_move((x, y), color=go.BLACK)
        self.assertTrue(gs.is_eye((0, 0), go.BLACK))


class TestGroups(unittest.TestCase):

    def test_liberties_after_capture(self):
        # creates 3x3 black group in the middle, that is then all captured
        # ...then an assertion is made that the resulting liberties after
        # capture are the same as if the group had never been there

        gs_capture = GameState(size=7)
        gs_reference = GameState(size=7)
        # add in 3x3 black stones
        for x in range(2, 5):
            for y in range(2, 5):
                gs_capture.do_move((x, y), go.BLACK)
        # surround the black group with white stones
        # and set the same white stones in gs_reference
        for x in range(2, 5):
            gs_capture.do_move((x, 1), go.WHITE)
            gs_capture.do_move((x, 5), go.WHITE)
            gs_reference.do_move((x, 1), go.WHITE)
            gs_reference.do_move((x, 5), go.WHITE)
        gs_capture.do_move((1, 1), go.WHITE)
        gs_reference.do_move((1, 1), go.WHITE)
        for y in range(2, 5):
            gs_capture.do_move((1, y), go.WHITE)
            gs_capture.do_move((5, y), go.WHITE)
            gs_reference.do_move((1, y), go.WHITE)
            gs_reference.do_move((5, y), go.WHITE)

        # board configuration and liberties of gs_capture and of gs_reference should be identical
        self.assertTrue(gs_reference.is_board_equal(gs_capture))
        self.assertTrue(gs_reference.is_liberty_equal(gs_capture))

    def test_large_group_neighbors(self):

        gs, _ = parseboard.parse(". . B B B . .|"
                                 ". . B B B . .|"
                                 ". . B B B . .|"
                                 ". . W W W . .|"
                                 ". . W W W . .|"
                                 ". . W W W . .|"
                                 ". . . . . . .|")
        self.assertTrue(gs.sanity_check_groups())


class TestCopy(unittest.TestCase):

    def equality_checks(self, original, copy):
        self.assertListEqual(copy.get_legal_moves(), original.get_legal_moves())
        self.assertListEqual(copy.get_history(), original.get_history())
        self.assertTrue(copy.is_board_equal(original))
        self.assertTrue(copy.is_liberty_equal(original))
        self.assertEqual(copy.get_hash(), original.get_hash())
        self.assertListEqual(copy.get_history(), original.get_history())
        self.assertEqual(copy.get_captures_white(), original.get_captures_white())
        self.assertEqual(copy.get_captures_black(), original.get_captures_black())

    def test_copy(self):
        gs, _ = parseboard.parse(". B . . . . .|"
                                 "B W W . . . .|"
                                 ". B W . B . .|"
                                 ". . . . . . B|"
                                 ". . B . . . .|"
                                 "W . . . W W .|")

        copy = gs.copy()

        self.assertTrue(copy.sanity_check_groups())
        self.equality_checks(gs, copy)


class TestTemporaryMove(unittest.TestCase):

    def listNotEqual(self, listA, listB):
        if len(listA) != len(listB):
            return True
        else:
            for (a, b) in zip(listA, listB):
                if a != b:
                    return True
            return False

    def equality_checks(self, original, copy):
        self.assertEqual(copy.get_current_player(), original.get_current_player())
        self.assertListEqual(copy.get_legal_moves(), original.get_legal_moves())
        self.assertListEqual(copy.get_history(), original.get_history())
        self.assertTrue(copy.is_board_equal(original))
        self.assertTrue(copy.is_liberty_equal(original))
        self.assertEqual(copy.get_hash(), original.get_hash())
        self.assertEqual(copy.get_captures_white(), original.get_captures_white())
        self.assertEqual(copy.get_captures_black(), original.get_captures_black())

    def inequality_checks(self, original, copy):
        self.assertTrue(self.listNotEqual(copy.get_legal_moves(), original.get_legal_moves()))
        self.assertTrue(self.listNotEqual(copy.get_history(), original.get_history()))
        self.assertFalse(copy.is_board_equal(original))
        self.assertFalse(copy.is_liberty_equal(original))
        self.assertNotEqual(copy.get_hash(), original.get_hash())

    def test_simple_undo(self):
        gs = GameState(size=7)
        copy = gs.copy()

        # Baseline equality checks between gs and copy
        self.equality_checks(gs, copy)

        with copy.try_stone(0):
            self.assertTrue(gs.sanity_check_groups())
            self.assertTrue(copy.sanity_check_groups())
            self.inequality_checks(gs, copy)

            # (0, 0) is occupied and should currently be illegal
            self.assertFalse(copy.is_legal((0, 0)))

        # Move should now be undone - retry equality checks from above
        self.assertTrue(gs.sanity_check_groups())
        self.assertTrue(copy.sanity_check_groups())
        self.equality_checks(gs, copy)

        # With move undone, it should be legal again
        self.assertTrue(copy.is_legal((0, 0)))

    def test_ko_undo(self):
        gs, moves = parseboard.parse(". B . . . . .|"
                                     "B W B . . . .|"
                                     "W k W . . . .|"
                                     ". W . . . . .|"
                                     ". . . . . . .|"
                                     ". . . . a . .|"
                                     ". . . . . . .|")
        gs.set_current_player(go.BLACK)

        # Trigger ko at (1, 1)
        gs.do_move(moves['k'])
        ko = gs.get_ko_location()
        self.assertIsNotNone(ko)

        copy = gs.copy()

        self.equality_checks(gs, copy)

        with copy.try_stone(flatten_idx(moves['a'], gs.get_size())):
            self.inequality_checks(gs, copy)

            # Doing move at 'a' clears ko
            self.assertIsNone(copy.get_ko_location())

        self.equality_checks(gs, copy)

        # Undoing move at 'a' resets ko
        self.assertEqual(copy.get_ko_location(), ko)

    def test_simple_merge_undo(self):
        gs, moves = parseboard.parse(". . . . . . .|"
                                     ". . . B W . .|"
                                     ". . . B W . .|"
                                     ". . . a W . .|"
                                     ". . . B W . .|"
                                     ". . . B W . .|"
                                     ". . . . . . .|")
        gs.set_current_player(go.BLACK)

        copy = gs.copy()

        # Initial equality checks
        self.assertTrue(copy.sanity_check_groups())
        self.equality_checks(gs, copy)

        with copy.try_stone(flatten_idx(moves['a'], gs.get_size())):
            self.assertTrue(copy.sanity_check_groups())
            self.inequality_checks(gs, copy)

        # Move should now be undone - retry equality checks from above
        self.assertTrue(copy.sanity_check_groups())
        self.equality_checks(gs, copy)

    def test_simple_capture_undo(self):
        gs, moves = parseboard.parse(". . . . . . .|"
                                     ". . . . . . .|"
                                     ". . . . B . .|"
                                     ". . . B W c .|"
                                     ". . . B W B .|"
                                     ". . . . B . .|"
                                     ". . . . . . .|")
        gs.set_current_player(go.BLACK)

        copy = gs.copy()

        # Initial equality checks
        self.assertTrue(copy.sanity_check_groups())
        self.equality_checks(gs, copy)

        with copy.try_stone(flatten_idx(moves['c'], gs.get_size())):
            self.assertTrue(copy.sanity_check_groups())
            self.inequality_checks(gs, copy)

        # Move should now be undone - retry equality checks from above
        self.assertTrue(copy.sanity_check_groups())
        self.equality_checks(gs, copy)

    def test_merge_and_capture_undo(self):
        gs, moves = parseboard.parse(". . B B B . .|"
                                     ". B W W W B .|"
                                     ". B W B W B .|"
                                     ". B W c W B .|"
                                     ". B W B W B .|"
                                     ". B W W W B .|"
                                     ". . B B B . .|")
        gs.set_current_player(go.BLACK)

        copy = gs.copy()

        # Initial equality checks
        self.assertTrue(copy.sanity_check_groups())
        self.equality_checks(gs, copy)

        with copy.try_stone(flatten_idx(moves['c'], gs.get_size())):
            self.assertTrue(copy.sanity_check_groups())
            self.inequality_checks(gs, copy)

        # Move should now be undone - retry equality checks from above
        self.assertTrue(copy.sanity_check_groups())
        self.equality_checks(gs, copy)

    def test_hash_update_matches_actual_hash(self):
        gs = GameState(size=7)
        gs, moves = parseboard.parse("a x b . . . .|"
                                     "z c d . . . .|"
                                     ". . . . . . .|"
                                     ". . . y . . .|"
                                     ". . . . . . .|"
                                     ". . . . . . .|"
                                     ". . . . . . .|")

        # a,b,c,d are black, x,y,z,x are white
        move_order = ['a', 'x', 'b', 'y', 'c', 'z', 'd', 'x']
        for m in move_order:
            move_1d = flatten_idx(moves[m], gs.get_size())

            # 'Try' move and get hash
            with gs.try_stone(move_1d):
                hash1 = gs.get_hash()

            # Actually do move and get hash
            gs.do_move(moves[m])
            hash2 = gs.get_hash()

            self.assertEqual(hash1, hash2)


if __name__ == '__main__':
    unittest.main()
