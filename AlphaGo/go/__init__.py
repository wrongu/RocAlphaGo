from .game_state import GameState, IllegalMove

# Expose constants to python (copied from AlphaGo/go/constants.pxd)
PASS = None
EMPTY = 2
WHITE = 3
BLACK = 4

__all__ = ['GameState', 'IllegalMove', 'PASS', 'BLACK', 'WHITE', 'EMPTY']
