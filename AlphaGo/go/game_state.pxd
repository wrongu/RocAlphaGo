from AlphaGo.go.constants cimport stone_t, group_t, action_t
from AlphaGo.go.coordinates cimport calculate_board_location, calculate_tuple_location, \
    get_pattern_hash, get_neighbors, get_3x3_neighbors, get_12d_neighbors
from AlphaGo.go.group_logic cimport Group, group_new, group_duplicate, group_add_stone, \
    group_merge, group_get_stone, group_remove_stone, group_add_liberty, group_remove_liberty
from AlphaGo.go.zobrist cimport get_zobrist_lookup, update_hash_by_location, update_hash_by_group
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
from libcpp.unordered_set cimport unordered_set as cpp_set
import numpy as np
cimport numpy as np


############################################################################
#   Typedefs                                                               #
#                                                                          #
############################################################################

ctypedef short location_t
ctypedef unsigned long pattern_hash_t
ctypedef unsigned long long zobrist_hash_t
ctypedef vector[location_t] pattern_t  # lookup of neighbor coordinates (or border)
ctypedef shared_ptr[Group] group_ptr_t  # smart pointer with reference counting wrapping a 'Group'
ctypedef cpp_set[group_ptr_t] group_set_t  # type for unordered set of unique groups
ctypedef vector[group_ptr_t] board_group_t  # type for group-lookup by board position


############################################################################
#   Class definition                                                       #
#                                                                          #
############################################################################

cdef class GameState:

    # Dimensions of one side of the board and total number of squares, respectively
    cdef short size, board_size

    # Possible ko location
    cdef location_t ko

    # Unordered list of all groups of stones
    cdef group_set_t groups_set

    # Lookup of a group from board location. Length is board size + 1 to include border
    cdef board_group_t board

    # Placeholder groups for referencing empty spaces or the border.
    cdef group_ptr_t group_empty, group_border

    # Current player and opponent, either _WHITE or _BLACK
    cdef stone_t current_player, opponent_player

    # Amount of black stones captured by white, and vice versa
    cdef short capture_black, capture_white

    # Amount of passes by black and by white, respectively
    cdef short passes_black, passes_white

    # List with move history
    cdef vector[location_t] moves_history

    # Number of handicap stones placed by BLACK at the start of the game
    cdef short num_handicap

    # List with legal moves
    cdef vector[location_t] legal_moves

    # Neighbors lookup tables. Pointers are to a single global instance of each variable shared by
    # all instances of GameState.
    cdef pattern_t* ptr_neighbor
    cdef pattern_t* ptr_neighbor3x3
    cdef pattern_t* ptr_neighbor12d

    # Zobrist hashing
    cdef zobrist_hash_t zobrist_current
    # Note: as with neighbor arrays, the zobrist lookup table is shared by all instances of
    # GameState, so it is stored with a pointer to the global instance.
    cdef vector[zobrist_hash_t]* ptr_zobrist_lookup

    cdef bool enforce_superko
    cdef set previous_hashes

    ############################################################################
    #   init functions                                                         #
    #                                                                          #
    ############################################################################

    cdef void initialize_new(self, short size, bool enforce_superko)
    """Initialize this state as empty state
    """

    cdef void initialize_duplicate(self, GameState copy_state)
    """Initialize all variables as a copy of copy_state
    """

    ############################################################################
    #   private cdef functions used for game-play                              #
    #                                                                          #
    ############################################################################

    cdef bool is_positional_superko(self, location_t location)
    """Find all actions that the current_player has done in the past.

       This takes into account the fact that history starts with BLACK when there are no handicaps
       or with WHITE when there are.
    """

    cpdef bool is_legal_move(self, location_t location)
    """Check if playing at location is a legal move
    """

    cdef bool has_liberty_after(self, location_t location)
    """Check if a play at location results in an alive group

       True if any of the following is true:
       - has liberty
       - connects to group with >= 2 liberty
       - captures enemy group
    """

    cdef void update_legal_moves(self)
    """Update legal_moves list
    """

    cdef void swap_players(self)
    """Switch current_player and opponent_player
    """

    cpdef void set_current_player(self, stone_t color)
    """Set current player to the given color.
    """

    ############################################################################
    #   private cdef helper functions for feature generation                   #
    #                                                                          #
    ############################################################################

    cdef bool is_eyeish(self, location_t location, stone_t owner)
    """Check if a location is 'eyeish'; that is, check that all 4 neighbors are either border
       or the same color.
    """

    cdef bool is_true_eye(self, location_t location, stone_t owner, list stack=*)
    """Check if location is a 'real' eye; this goes beyond checking if a location is 'eyeish' by
       checking that corners have the same owner or (recursively) are themselves eyes.
    """

    ############################################################################
    #   public cdef functions used by preprocessing                            #
    #                                                                          #
    ############################################################################

    cdef pattern_hash_t get_12d_hash(self, location_t center, bool include_player, int max_liberty=*)  # noqa:E501
    """Get unique-ish hash of the 12-stone pattern centered at 'center'. Assumes 'center'
       itself is EMPTY. If 'include_player' is True, hash also takes into account who is the
       current player.
    """

    cdef pattern_hash_t get_3x3_hash(self, short center, bool include_player, int max_liberty=*)
    """Get unique-ish hash of the 8-stone pattern centered at 'center'. Assumes 'center' itself
       is EMPTY. If 'include_player' is True, hash also takes into account who is the current
       player.
    """

    cdef vector[location_t] get_sensible_moves(self)
    """'Sensible' moves are all legal moves that are not eyes of the current player.
    """

    ############################################################################
    #   Helper functions for managing groups                                   #
    #                                                                          #
    ############################################################################

    cdef group_ptr_t combine_groups(self, group_ptr_t group_keep, group_ptr_t group_remove)
    """Combine group_keep and group_remove by copying all stones and liberties from
       group_remove into group_keep, and update pointers on the board.

       Returns group_keep
    """

    cdef void remove_group(self, group_ptr_t group_remove)
    """Remove group from everywhere in the state.
    """

    ############################################################################
    #   public cpdef functions used for game play                              #
    #                                                                          #
    ############################################################################

    cpdef void print_groups(self)
    """Debugging helper: prints address of all groups in this state and info about each one.
    """

    cpdef bool sanity_check_groups(self)
    """Debugging helper: loops over every location and group on the board and checks that they are
       self-consistent.
    """

    cdef location_t add_stone(self, location_t location)
    """Play stone on location and update the state. MOVE MUST BE LEGAL. Returns new ko location,
       or -1 if there is none.
    """

    cpdef TemporaryMove try_stone(self, location_t location, bool prepare_next=*)
    """Analogous to add_stone() for use in a with-statement. Automatically undoes the given move
       when the with-statement exits.

       For example:

           state.add_stone(loc1)
           with state.try_stone(loc2):
               print state.get_history()[-1]  # prints loc2
           # Here, state is returned to its value to before the with statement.
           print state.get_history()[-1]  # prints loc1
    """

    cpdef list get_legal_moves(self, bool include_eyes=*)
    """Return a list with all legal moves (in/excluding eyes)
    """

    cpdef float get_score(self, float komi=*)
    """Calculate score of board state. Uses 'Area scoring'.

       http://senseis.xmp.net/?Passing#1

       negative value indicates black win
       positive value indicates white win
    """

    cpdef stone_t get_winner_color(self, float komi=*)
    """Calculate score of board state and return player ID (1, -1, or 0 for tie)
       corresponding to winner. Uses 'Area scoring'.

       http://senseis.xmp.net/?Passing#1
    """

    cpdef void do_move(self, tuple action, stone_t color=*)
    """Play stone at action=(x,y). Use action=_PASS (-1) to pass. Checks move legality first.

       If it is a legal move, current_player switches to the opposite color. If not, an
       IllegalMove exception is raised
    """

    cpdef void place_handicap_stone(self, tuple action, stone_t color=*)
    """Add handicap stones given by a list of tuples in list handicap
    """

    cpdef void place_handicaps(self, list handicap)
    """Place list of handicap stones for BLACK (must be on empty board)
    """

    ############################################################################
    #   Helper functions for managing groups                                   #
    #                                                                          #
    ############################################################################

    cdef group_ptr_t combine_groups(self, group_ptr_t group_keep, group_ptr_t group_remove)
    """Combine group_keep and group_remove by copying all stones and liberties from
       group_remove into group_keep, and update pointers on the board.

       Returns group_keep
    """


cdef class TemporaryMove:
    """Helper-class for GameState.try_stone()

       This class implements python's 'with' interface such that when the 'with' block is exited,
       the move is undone.

       See https://www.python.org/dev/peps/pep-0343/ for more information about how 'with' interacts
       with __enter__ and __exit__
    """

    # Reference to the GameState being modified
    cdef GameState state

    # Current player / owner of the new stone
    cdef stone_t player_color

    # Where the stone is played
    cdef location_t move

    # If there was a ko location previously
    cdef location_t previous_ko

    # Reference to each of the up-to-4 opponent/friendly groups of the new stone before stone is
    # played so they can be restored regardless of merges/captures.
    cdef group_set_t neighbors_opponent, neighbors_friendly

    # Boolean flag indicating whether to prepare for another call to state.try_stone(). That is, if
    # this flag is True, it update history and switches to the next player. If it is False, it
    # simply adds the stone and makes no further updates.
    cdef bool prepare_next
