from AlphaGo.go.constants cimport stone_t, group_t, action_t
from AlphaGo.go.game_state cimport GameState
from AlphaGo.go.group_logic cimport Group, group_new, group_add_stone, group_add_liberty, \
    group_remove_liberty, group_merge, group_lookup
from AlphaGo.go.coordinates cimport get_pattern_hash
from AlphaGo.go.ladders cimport is_ladder_escape_move, is_ladder_capture_move, \
    get_plausible_escape_moves, get_plausible_capture_moves
from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np


############################################################################
#   Typedefs                                                               #
#                                                                          #
############################################################################

ctypedef short location_t
ctypedef shared_ptr[Group] group_ptr_t


# Feature tensors are 1-hot; using small 8-bit integers for tensor type.
ctypedef np.uint8_t onehot_t

# 'Lookahead' has features ranging between 0 and board_size. Use int16 for this type.
ctypedef np.uint16_t lookahead_t

# Type defining cdef function handle.
ctypedef int (*preprocess_method)(Preprocess, GameState, onehot_t[:, :], lookahead_t[:, :], int)


############################################################################
#   Class definition                                                       #
#                                                                          #
############################################################################

cdef class Preprocess:

    # All feature processors
    cdef vector[preprocess_method] processors

    # Flag whether or not any features require 'lookahead' to groups_after
    cdef bool requires_groups_after

    # List with all string names of features used currently
    cdef list feature_list

    # Output tensor size
    cdef int output_dim

    # Board size, same convention as in GameState
    cdef short size, board_size

    # Sets of pattern hashes of the Top-N most common response patterns of each type
    cdef set pattern_nakade
    cdef set pattern_response_12d
    cdef set pattern_non_response_3x3

    ############################################################################
    #   Feature-generating functions                                           #
    #                                                                          #
    ############################################################################

    cdef int get_board(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset)  # noqa: E501
    """A feature encoding _WHITE _BLACK and _EMPTY on separate planes.

       Note:
       - plane 0 always refers to the current player stones
       - plane 1 to the opponent stones
       - plane 2 to empty locations
    """

    cdef int get_turns_since(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset)  # noqa: E501
    """A feature encoding the age of the stone at each location up to 'maximum'

       Note:
       - the [maximum-1] plane is used for any stone with age greater than or equal to maximum
       - _EMPTY locations are all-zero features
    """

    cdef int get_liberties(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset)  # noqa: E501
    """A feature encoding the number of liberties of the group connected to the stone at each
       location

       Note:
       - there is no zero-liberties plane; the 0th plane indicates groups in atari
       - the [maximum-1] plane is used for any stone with liberties greater than or equal to maximum
       - _EMPTY locations are all-zero features
    """

    cdef int get_capture_size(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset)  # noqa: E501
    """A feature encoding the number of opponent stones that would be captured by playing at each
       location, up to 'maximum'

       Note:
       - we currently *do* treat the 0th plane as "capturing zero stones"
       - the [maximum-1] plane is used for any capturable group of size greater than or equal to
         maximum-1
       - the 0th plane is used for legal moves that would not result in capture
       - illegal move locations are all-zero features
    """

    cdef int get_self_atari_size(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset)  # noqa: E501
    """A feature encoding the size of the own-stone group that is put into atari by playing at a
       location
    """

    cdef int get_liberties_after(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset)  # noqa: E501
    """A feature encoding what the number of liberties *would be* of the group connected to the
       stone *if* played at a location

       Note:
       - there is no zero-liberties plane; the 0th plane indicates groups in atari
       - the [maximum-1] plane is used for any stone with liberties greater than or equal to maximum
       - illegal move locations are all-zero features
    """

    cdef int get_ladder_capture(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset)  # noqa: E501
    """A feature wrapping GameState.is_ladder_capture(). Check if an opponent group can be captured
       in a ladder
    """

    cdef int get_ladder_escape(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset)  # noqa: E501
    """A feature wrapping GameState.is_ladder_escape(). Check if current_player group can escape
       ladder
    """

    cdef int get_sensibleness(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset)  # noqa: E501
    """A move is 'sensible' if it is legal and if it does not fill the current_player's own eye
    """

    cdef int get_legal(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset)  # noqa: E501
    """Zero at all illegal moves, one at all legal moves. Unlike sensibleness, no eyes check.
    """

    cdef int zeros(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset)  # noqa: E501
    """Plane filled with zeros
    """

    cdef int ones(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset)  # noqa: E501
    """Plane filled with ones
    """

    cdef int color(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset)  # noqa: E501
    """Value net feature, plane with ones if active_player is black else zeros
    """

    cdef int ko(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset)  # noqa: E501
    """Ko positions
    """

    cdef int get_response(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset)  # noqa:E501
    """Single feature plane encoding whether this location matches any of the response
       patterns, for now it only checks the 12d response patterns as we do not use the
       3x3 response patterns.
    """

    cdef int get_save_atari(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset)  # noqa:E501
    """A feature wrapping GameState.is_ladder_escape().
       check if current_player group can escape atari for at least one turn
    """

    cdef int get_neighbor(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset)  # noqa:E501
    """Encode last move neighbor positions in two planes:
       - horizontal & vertical / direct neighbor
       - diagonal neighbor
    """

    cdef int get_nakade(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset)  # noqa:E501
    """A nakade pattern is a 12d pattern on a location a stone was captured before
       it is unclear if a max size of the captured group has to be considered and
       how recent the capture event should have been

       the 12d pattern can be encoded without stone color and liberty count
       unclear if a border location should be considered a stone or liberty

       pattern lookup value is being set instead of 1
    """

    cdef int get_response_12d(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset)  # noqa:E501
    """Set 12d hash pattern for 12d shape around last move
       pattern lookup value is being set instead of 1
    """

    cdef int get_non_response_3x3(self, GameState state, onehot_t[:, :] tensor, lookahead_t[:, :] groups_after, int offset)  # noqa:E501
    """Set 3x3 hash pattern for every legal location where
       pattern lookup value is being set instead of 1
    """

    ############################################################################
    #   public cdef function                                                   #
    #                                                                          #
    ############################################################################

    cpdef np.ndarray[onehot_t, ndim=4] state_to_tensor(self, GameState state)
    """Convert a GameState to a Theano-compatible tensor of one-hot features
    """

############################################################################
#   "Groups after" helper functions for lookahead without copying state    #
#                                                                          #
############################################################################

cdef np.ndarray[lookahead_t, ndim=2] get_groups_after(GameState state)
"""Without creating a copy of the state, compute features of the resulting board state
   IF a stone were played at each legal location. Three features are computed and stored in
   a board_size x 3 array. Values are only computed at legal move locations.

   That is, groups_after[loc, _STONE] contains the size of the group that would be formed
   if the current_player were to play a stone at 'loc', hence the name groups_after

   Returns a numpy array of size (board_size, 3)
   - groups_after[loc, 0] = resulting group size by playing at loc
   - groups_after[loc, 1] = number of remaining liberties of group by playing at loc
   - groups_after[loc, 2] = number of stones captured by playing at loc
"""

cdef np.ndarray[lookahead_t, ndim=1] get_groups_after_at(GameState state, location_t loc)
"""Compute 'groups_after' results at a single location, which must be a legal move.

   Returns a size (3,) numpy arry with group size in index 0, liberty count in index 1, and
   number of opponent stones captured in index 2 (see get_groups_after())
"""
