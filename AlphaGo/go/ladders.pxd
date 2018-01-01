from AlphaGo.go.constants cimport stone_t, group_t
from AlphaGo.go.group_logic cimport Group, group_get_liberty, group_get_stone
from AlphaGo.go.game_state cimport GameState
from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector


ctypedef short location_t
ctypedef shared_ptr[Group] group_ptr_t  # smart pointer with reference counting wrapping a 'Group'
ctypedef vector[group_ptr_t] board_group_t  # type for group-lookup by board position
ctypedef vector[location_t] pattern_t  # lookup of neighbor coordinates (or border)


cdef bool is_ladder_escape_move(GameState copy_state, group_ptr_t prey, location_t move, int depth=*)  # noqa:E501
"""(Inefficiently) check whether the given move escapes ladder capture of the given group.
   Returns True when escape is possible, or recursion depth limit is reached (assuming that the
   opponent does not recognize ladders with greater depth as a 'capture' either)

   Preconditions:
   - GameState 'copy_state' is safe to be altered
   - prey group is in atari and owned by copy_state.current_player
   - given move is legal
   - depth >= 0

   Note: while optimizations can be made, this version is easy to understand, and ladders are
   less likely to be used as features in a production computer-go system.
"""

cdef bool is_ladder_capture_move(GameState copy_state, group_ptr_t prey, location_t move, int depth=*)  # noqa:E501
"""(Inefficiently) check whether the given move captures the prey, or forces capture of the
   prey by a ladder within 'depth' moves.

   Preconditions:
   - GameState 'copy_state' is safe to be altered
   - prey group has <= 2 liberties and is owned by the opponent
   - given move is legal
   - depth >= 0

   Note: while optimizations can be made, this version is easy to understand, and ladders are
   less likely to be used as features in a production computer-go system.
"""

cdef set get_plausible_escape_moves(GameState state, group_ptr_t prey)
"""Get set of moves that should be checked as possible escape moves for the given prey.
"""

cdef set get_plausible_capture_moves(GameState state, group_ptr_t prey)
"""Get set of moves that should be checked as possible escape moves for the given prey.
"""

cdef set get_adjacent_captures(group_ptr_t group, board_group_t &board, pattern_t &neighbor_lookup)
"""Search for moves that the owner of 'group' could play that would kill opponent groups
   adjacent to it.

   Note: this would (probably) consitute an escape from a ladder if 'group' is in atari.
"""
