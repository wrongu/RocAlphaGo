import unittest
from AlphaGo import go
from multiprocessing import Process
from interface.gtp_wrapper import run_gtp


class PassPlayer(object):
    def get_move(self, state):
        return go.PASS


class TestGTPProcess(unittest.TestCase):

    def test_run_commands(self):
        def stdin_simulator():
            return "\n".join([
                "1 name",
                "2 boardsize 19",
                "3 clear_board",
                "4 genmove black",
                "5 genmove white",
                "99 quit"])

        gtp_proc = Process(target=run_gtp, args=(PassPlayer(), stdin_simulator))
        gtp_proc.start()
        gtp_proc.join(timeout=1)


if __name__ == '__main__':
    unittest.main()
