# cython: wraparound=False
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
from AlphaGo.util import unflatten_idx
from cython.operator cimport dereference as d


cdef bool is_ladder_escape_move(GameState state, group_ptr_t prey, location_t move, int depth=50):  # noqa:E501
    """(Inefficiently) check whether the given move escapes ladder capture of the given group.
       Returns True when escape is plausible, or recursion depth limit is reached (assuming that the
       opponent does not recognize ladders with greater depth as a 'capture' either)

       Preconditions:
       - GameState 'state' is safe to be temporarily altered (not thread-safe!)
       - prey group is in atari and owned by state.current_player
       - given move is legal
       - depth >= 0

       Note: while optimizations can be made by avoiding copied states, this version is at least
       easily comprehensible.
    """

    cdef location_t prey_loc = group_get_stone(prey)

    # Base case: if search depth is exhausted, assume that the ladder is escable.
    if depth <= 0:
        return True

    # Try move and check results.
    with state.try_stone(move):
        prey = state.board[prey_loc]

        # Case 1: prey has >= 3 liberties after move, in which case it escaped.
        if d(prey).count_liberty >= 3:
            return True

        # Case 2: prey is left in atari, in which case it did not escape.
        elif d(prey).count_liberty == 1:
            return False

        # Case 3: prey has 2 liberties left, in which case it may still be captured in a ladder.
        # Requires recursive search.
        else:
            # Opponent may attempt to capture at either of the prey's two liberties.
            for plausible_capture in get_plausible_capture_moves(state, prey):
                if is_ladder_capture_move(state.copy(), prey, plausible_capture, depth - 1):
                    return False

            # If reached here, none of prey's liberties are ladder captures, in which case it
            # escaped.
            return True

cdef bool is_ladder_capture_move(GameState state, group_ptr_t prey, location_t move, int depth=50):  # noqa:E501
    """(Inefficiently) check whether the given move captures the prey, or forces capture of the
       prey by a ladder within 'depth' moves.

       Preconditions:
       - GameState 'state' is safe to be temporarily altered (not thread-safe!)
       - prey group has <= 2 liberties and is owned by the opponent
       - given move is legal
       - depth >= 0

       Note: while optimizations can be made by avoiding copied states, this version is at least
       easily comprehensible.
    """

    cdef location_t prey_loc = group_get_stone(prey)

    # Base case: if search depth is exhausted, assume that ladder is not capturable.
    if depth <= 0:
        return False

    # Try the move and check results
    with state.try_stone(move):
        prey = state.board[prey_loc]

        # Case 1: prey has >= 2 liberties after move, in which case it escaped.
        if d(prey).count_liberty >= 2:
            return False

        # Case 2: prey has 1 liberty after move, in which case it may still attempt to escape.
        # Requires recursive search.
        elif d(prey).count_liberty == 1:
            # Try each potential escape move
            for plausible_escape in get_plausible_escape_moves(state, prey):
                if is_ladder_escape_move(state.copy(), prey, plausible_escape, depth - 1):
                    return False

        # If reached here, either prey was captured or no escape move was found
        return True

cdef set get_plausible_escape_moves(GameState state, group_ptr_t prey):
    """Get set of moves that should be checked as plausible escape moves for the given prey.

       Preconditions:
       - current_player is prey owner
       - prey is in atari
    """

    cdef set to_remove = set(), plausible_escapes = set()
    cdef location_t move
    cdef stone_t owner = d(prey).color
    cdef bool is_sensible

    # Plausible escapes 1: remaining liberty of the prey group
    plausible_escapes.add(group_get_liberty(prey))

    # Plausible escapes 2: any moves that capture a group adjacent to the prey (since this would
    # free up new liberties)
    plausible_escapes.update(get_adjacent_captures(prey, state.board, d(state.ptr_neighbor)))

    # Ensure that all moves are not only legal, but 'sensible'.
    for move in plausible_escapes:
        is_sensible = state.is_legal_move(move) and not state.is_true_eye(move, owner)
        if not is_sensible:
            to_remove.add(move)
    plausible_escapes.difference_update(to_remove)

    return plausible_escapes

cdef set get_plausible_capture_moves(GameState state, group_ptr_t prey):
    """Get set of moves that should be checked as plausible escape moves for the given prey.

       Preconditions:
       - current_player is opponent of prey
       - prey has exactly 2 liberties
    """

    cdef set to_remove = set(), plausible_captures = set()
    cdef location_t loc
    cdef group_t val

    for loc, val in d(prey).locations:
        if val == group_t.LIBERTY:
            plausible_captures.add(loc)

    # Ensure that all moves are legal (note that no 'sensibility' check is needed here since a move
    # cannot be both a liberty of the other player and an eye of the current player).
    for move in plausible_captures:
        if not state.is_legal_move(move):
            to_remove.add(move)
    plausible_captures.difference_update(to_remove)

    return plausible_captures

cdef set get_adjacent_captures(group_ptr_t group, board_group_t &board, pattern_t &neighbor_lookup):
    """Search for moves that the owner of 'group' could play that would kill opponent groups
       adjacent to it. This would consitute an escape from a ladder if 'group' is in atari.
    """

    # 'atari_liberties' holds liberty locations of enemy groups that are adjacent to the given group
    # 'and are in atari (hence could be captured by playing at the given location)
    cdef set atari_liberties = set()
    cdef group_ptr_t neighbor_group
    cdef location_t loc, opp_loc, neighbor_loc
    cdef group_t val, opp_val
    cdef stone_t owner = d(group).color
    cdef int i

    for loc, val in d(group).locations:
        if val == group_t.STONE:
            for i in range(4):
                neighbor_loc = neighbor_lookup[loc * 4 + i]
                neighbor_group = board[neighbor_loc]

                # Check if neighbor of this group stone is opponent group.
                if d(neighbor_group).color > stone_t.EMPTY and d(neighbor_group).color != owner:
                    # Further check if opponent group is in atari and can be captured.
                    if d(neighbor_group).count_liberty == 1:
                        # Find the one liberty of this group and mark it as an escape move.
                        atari_liberties.add(group_get_liberty(neighbor_group))
    return atari_liberties
