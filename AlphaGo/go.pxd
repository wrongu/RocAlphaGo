import numpy as np
cimport numpy as np
from AlphaGo.go_data cimport *


cdef class GameState:

    ############################################################################
    #   variables declarations                                                 #
    #                                                                          #
    ############################################################################

    # amount of locations on one side
    cdef char  size
    # amount of locations on board, size * size
    cdef short board_size

    # possible ko location
    cdef short ko

    # list with all groups
    cdef Groups_List *groups_list
    # pointer to empty group
    cdef Group *group_empty

    # list representing board locations as groups
    # a Group contains all group stone locations and group liberty locations
    cdef Group **board_groups

    cdef char player_current
    cdef char player_opponent

    # amount of black stones captured
    cdef short capture_black
    # amount of white stones captured
    cdef short capture_white

    # amount of passes by black
    cdef short passes_black
    # amount of passes by white
    cdef short passes_white

    # list with move history
    cdef Locations_List *moves_history

    # list with legal moves
    cdef Locations_List *moves_legal

    # arrays, neighbor arrays pointers
    cdef short *neighbor
    cdef short *neighbor3x3
    cdef short *neighbor12d

    # zobrist
    cdef unsigned long long zobrist_current
    cdef unsigned long long *zobrist_lookup

    cdef bint enforce_superko
    cdef set previous_hashes

    ############################################################################
    #   init functions                                                         #
    #                                                                          #
    ############################################################################

    cdef void initialize_new(self, char size, bint enforce_superko)
    """Initialize this state as empty state
    """

    cdef void initialize_duplicate(self, GameState copyState)
    """Initialize all variables as a copy of copy_state
    """

    ############################################################################
    #   private cdef functions used for game-play                              #
    #                                                                          #
    ############################################################################

    cdef void update_hash(self, short location, char colour)
    """xor current hash with location + colour action value
    """

    cdef bint is_positional_superko(self, short location, Group **board)
    """Find all actions that the current_player has done in the past.

    This takes into account the fact that history starts with BLACK when there are no handicaps or
    with WHITE when there are.
    """

    cdef bint is_legal_move(self, short location, Group **board, short ko)
    """Check if playing at location is a legal move
    """

    cdef bint is_legal_move_superko(self, short location, Group **board, short ko)
    """Check if playing at location is a legal move, taking superko into account
    """

    cdef bint has_liberty_after(self, short location, Group **board)
    """Check if a play at location results in an alive group

    True if any of the following is true:
    - has liberty
    - connects to group with >= 2 liberty
    - captures enemy group
    """

    cdef short calculate_board_location(self, char x, char y)
    """2D tuple location to 1d index. Inverse of calculate_tuple_location()

    No sanity checks on bounds.

    - x is column
    - y is row
    """

    cdef tuple calculate_tuple_location(self, short location)
    """1d index to 2d tuple location. Inverse of calculate_board_location()

    No sanity checks on bounds.
    """

    cdef void set_moves_legal_list(self, Locations_List *moves_legal)
    """Generate moves_legal list
    """

    cdef void combine_groups(self, Group* group_keep, Group* group_remove, Group **board)
    """Combine group_keep and group_remove and replace group_remove on the board
    """

    cdef void remove_group(self, Group* group_remove, Group **board, short* ko)
    """Remove group from board -> set all locations to group_empty
    """

    cdef void add_to_group(self, short location, Group **board, short* ko, short* count_captures)
    """Check if a stone on location is connected to a group, kills a group or is a new group on the
    board
    """

    ############################################################################
    #   private cdef functions used for feature generation                     #
    #                                                                          #
    ############################################################################

    cdef long generate_12d_hash(self, short centre)
    """Generate 12d hash around centre location
    """

    cdef long generate_3x3_hash(self, short centre)
    """Generate 3x3 hash around centre location
    """

    cdef void get_group_after_pointer(self, short* stones, short* liberty, short* capture, char* locations, char* captures, short location)  # noqa: E501
    cdef void get_group_after(self, char* groups_after, char* locations, char* captures, short location)  # noqa: E501
    """Groups_after is a board_size * 3 array representing STONES, LIBERTY, CAPTURE for every location

       calculate group after a play on location and set
       groups_after[location * 3 +  ] to stone   count
       groups_after[location * 3 + 1] to liberty count
       groups_after[location * 3 + 2] to capture count
    """

    cdef bint is_true_eye(self, short location, Locations_List* eyes, char owner)
    """Check if location is a real eye
    """

    ############################################################################
    #   private cdef Ladder functions                                          #
    #                                                                          #
    ############################################################################

    """Ladder evaluation consumes a lot of time duplicating data, the original
       version (still can be found in go_python.py) made a copy of the whole
       GameState for every move played.

       This version only duplicates self.board_groups (so the list with pointers to groups)
       the add_ladder_move playes a move like the add_to_group function but it
       does not change the original groups and creates a list with groups removed

       with this groups removed list undo_ladder_move will return the board state to
       be the same as before add_ladder_move was called

       get_removed_groups and unremove_group are being used my add/undo_ladder_move

       nb.
       duplicating self.board_groups is not neccisary stricktly speaking but
       it is safer to do so in a threaded environment. as soon as mcts is
       implemented this duplication could be removed if the mcts ensures a
       GameState is not accesed while preforming a ladder evaluation

       TODO validate no changes are being made!

       TODO self.player colour is used, should become a pointer
    """

    cdef Groups_List* add_ladder_move(self, short location, Group **board, short* ko)
    """Create a new group for location move and add all connected groups to it

       similar to add_to_group except no groups are changed or killed and a list
       with groups removed is returned so the board can be restored to original
       position
    """

    cdef void remove_ladder_group(self, Group* group_remove, Group **board, short* ko)
    """Remove group from board -> set all locations to group_empty
       does not update zobrist hash
    """

    cdef void undo_ladder_move(self, short location, Groups_List* removed_groups, short ko, Group **board, short* ko)  # noqa: E501
    """Use removed_groups list to return board state to be the same as before
       add_ladder_move was used
    """

    cdef void unremove_group(self, Group* group_remove, Group **board)
    """Unremove group from board
       loop over all stones in this group and set board to group_unremove
       remove liberty from neigbor locations
    """

    cdef dict get_capture_moves(self, Group* group, char color, Group **board)
    """Create a dict with al moves that capture a group surrounding group
    """

    cdef void get_removed_groups(self, short location, Groups_List* removed_groups, Group **board, short* ko)  # noqa: E501
    """Create a new group for location move and add all connected groups to it

       similar to add_to_group except no groups are changed or killed
       all changes to the board are stored in removed_groups
    """

    cdef bint is_ladder_escape_move(self, Group **board, short* ko, Locations_List *list_ko, short location_group, dict capture, short location, int maxDepth, char colour_group, char colour_chase)  # noqa: E501
    """Play a ladder move on location, check if group has escaped,
       if the group has 2 liberty it is undetermined ->
       try to capture it by playing at both liberty
    """

    cdef bint is_ladder_capture_move(self, Group **board, short* ko, Locations_List *list_ko, short location_group, dict capture, short location, int maxDepth, char colour_group, char colour_chase)  # noqa: E501
    """Play a ladder move on location, try capture and escape moves
       and see if the group is able to escape ladder
    """

    ############################################################################
    #   public cdef functions used by preprocessing                            #
    #                                                                          #
    ############################################################################

    cdef char* get_groups_after(self)
    """Return a short array of size board_size * 3 representing
       STONES, LIBERTY, CAPTURE for every board location

       max count values are 100

       loop over all legal moves and determine stone count, liberty count and
       capture count of a play on that location
    """

    cdef long get_hash_12d(self, short centre)
    """Return hash for 12d star pattern around location
    """

    cdef long get_hash_3x3(self, short location)
    """Return 3x3 pattern hash + current player
    """

    cdef char* get_ladder_escapes(self, int maxDepth)
    """Return char array with size board_size
       every location represents a location on the board where:
       _FREE  = no ladder escape
       _STONE = ladder escape
    """

    cdef char* get_ladder_captures(self, int maxDepth)
    """Return char array with size board_size
       every location represents a location on the board where:
       _FREE  = no ladder capture
       _STONE = ladder capture
    """

    ############################################################################
    #   public cdef functions used for game play                               #
    #                                                                          #
    ############################################################################

    cdef void add_move(self, short location)
    """!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                  Move should be legal!
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

       play move on location, move should be legal!

       update player_current, history and moves_legal
    """

    cdef GameState new_state_add_move(self, short location)
    """Copy this gamestate and play move at location
    """

    cdef float get_score(self, float komi)
    """Calculate score of board state. Uses 'Area scoring'.

       http://senseis.xmp.net/?Passing#1

       negative value indicates black win
       positive value indicates white win
    """

    cdef char get_winner_colour(self, float komi)
    """Calculate score of board state and return player ID (1, -1, or 0 for tie)
       corresponding to winner. Uses 'Area scoring'.

       http://senseis.xmp.net/?Passing#1
    """

    ############################################################################
    #   public def functions used for game play (Python)                       #
    #                                                                          #
    ############################################################################

    cdef Locations_List* get_sensible_moves(self)
    """Only used for def get_legal_moves
    """
