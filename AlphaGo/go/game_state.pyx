# cython: wraparound=False
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
from cython.operator cimport dereference as d
import numpy as np
cimport numpy as np


############################################################################
#   Global variables shared by instances of GameState                      #
#                                                                          #
############################################################################

# Global arrays (lookup tables) for neighbor indices
cdef pattern_t neighbor
cdef pattern_t neighbor3x3
cdef pattern_t neighbor12d
cdef short neighbor_size

# Global array for zobrist lookup
cdef vector[zobrist_hash_t] zobrist_lookup


# Constant value used to generate pattern hashes (TODO: move elsewhere)
cdef int _HASHVALUE = 33


############################################################################
#   Class definition                                                       #
#                                                                          #
############################################################################


cdef class GameState:
    """Class representing the current state of a game of Go and its history.

       Note that the python interface passes moves as (x, y) tuples, while all cython code uses the
       slightly more streamlined 1D indexing.
    """

    ############################################################################
    #   all variables are declared in the .pxd file                            #
    #                                                                          #
    ############################################################################

    """
    # Dimensions of one side of the board and total number of squares, respectively
    cdef short size, board_size

    # Possible ko location
    cdef location_t ko

    # Unordered list of all groups of stones
    cdef group_set_t groups_set

    # Lookup of a group from board location. Length is board size + 1 to include border
    cdef board_group_t board

    # Current player and opponent, either WHITE or BLACK
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

    # Neighbors lookup tables
    cdef pattern_t* neighbor
    cdef pattern_t* neighbor3x3
    cdef pattern_t* neighbor12d

    # Zobrist hashing
    cdef zobrist_hash_t zobrist_current
    cdef vector[zobrist_hash_t] ptr_zobrist_lookup

    cdef bool enforce_superko
    cdef set previous_hashes
    """

    ############################################################################
    #   init functions                                                         #
    #                                                                          #
    ############################################################################

    cdef void initialize_new(self, short size, bool enforce_superko):
        """Initialize as a new state.
        """

        cdef short i

        # Initialize size and board_size
        self.size = size
        self.board_size = size * size

        # Create placeholder groups for empty spaces and border.
        self.group_empty = group_new(stone_t.EMPTY)
        self.group_border = group_new(stone_t.BORDER)

        # Create empty history list
        self.moves_history = vector[location_t]()

        # Initialize player colors
        self.current_player = stone_t.BLACK
        self.opponent_player = stone_t.WHITE

        self.ko = -1
        self.capture_black, self.capture_white = 0, 0
        self.passes_black, self.passes_white = 0, 0
        self.num_handicap = 0

        # 'board' represents all stones on the board by first pointing to all groups of stones.
        # Every group contains color, stone-locations and liberty locations. Border location is
        # included, therefore the array size is board_size + 1
        self.board = board_group_t(self.board_size + 1)
        self.board[self.board_size] = self.group_border

        # Create list of legal moves (initially everything is legal). This is updated after each
        # move.
        self.legal_moves = vector[location_t](self.board_size)

        # Create empty list of all current groups.
        self.groups_set = group_set_t()

        # Initialize board, set all locations to empty and populate list of legal moves.
        for i in range(self.board_size):
            self.board[i] = self.group_empty
            self.legal_moves[i] = i

        # Initialize zobrist hashing things.
        self.previous_hashes = set()
        self.zobrist_current = 0
        self.enforce_superko = enforce_superko

    cdef void initialize_duplicate(self, GameState copy_state):
        """Initialize all variables as a deep copy of copy_state
        """

        cdef int i
        cdef location_t loc
        cdef group_t val
        cdef group_ptr_t ref_group, dup_group

        # Copy each numeric instance variable's value.
        self.size = copy_state.size
        self.board_size = copy_state.board_size
        self.ko = copy_state.ko
        self.current_player = copy_state.current_player
        self.opponent_player = copy_state.opponent_player
        self.capture_black = copy_state.capture_black
        self.capture_white = copy_state.capture_white
        self.passes_black = copy_state.passes_black
        self.passes_white = copy_state.passes_white
        self.num_handicap = copy_state.num_handicap
        self.zobrist_current = copy_state.zobrist_current
        self.enforce_superko = copy_state.enforce_superko

        # Copy set of hashes using built in set.copy()
        self.previous_hashes = copy_state.previous_hashes.copy()

        # Copy all 'simple' C++ objects (i.e. non-pointers / non-groups) using default C++ copying
        # rules, which automatically does a deep-copy of containers.
        self.moves_history = copy_state.moves_history
        self.legal_moves = copy_state.legal_moves

        # Note: group_empty and group_border are constant, so duplicating the underlying object is
        # unnecessary.
        self.group_empty = copy_state.group_empty
        self.group_border = copy_state.group_border

        # Copy groups by duplicating the underlying groups objects.
        self.groups_set = group_set_t()
        self.board = board_group_t(self.board_size + 1)

        # Initialize all of self.board to the empty group, and set the border.
        for i in range(self.board_size):
            self.board[i] = self.group_empty
        self.board[self.board_size] = self.group_border

        # Loop over each unique group in copy_state and make a duplicate.
        for ref_group in copy_state.groups_set:
            dup_group = group_duplicate(ref_group)
            self.groups_set.insert(dup_group)

            # Set self.board[loc] for each stone in this group.
            for loc, val in d(dup_group).locations:
                if val == group_t.STONE:
                    self.board[loc] = dup_group

    def __init__(self, char size=19, GameState copy=None, enforce_superko=True):
        """Create new instance of GameState. If copy is supplied, creates a deep copy of another
           state. Otherwise, creates an empty state.
        """
        global neighbor, neighbor3x3, neighbor12d, zobrist_lookup, neighbor_size

        if copy is not None:
            size = copy.size

        # Check if this is the first GameState object (of this size) and initialize globals.
        if neighbor_size == 0 or neighbor_size != size:
            # Initialize "neighbor" lookup tables
            neighbor = get_neighbors(size)
            neighbor3x3 = get_3x3_neighbors(size)
            neighbor12d = get_12d_neighbors(size)
            zobrist_lookup = get_zobrist_lookup(size)

            # Set global size to detect whether globals need to be reinitialized for other GameState
            # instances with different sizes (which is unlikely).
            # TODO - global map from size to lookup table
            neighbor_size = size

        # Regardless of 'new' or 'duplicate', set pointers to global lookup tables and set size.
        self.ptr_neighbor = &neighbor
        self.ptr_neighbor3x3 = &neighbor3x3
        self.ptr_neighbor12d = &neighbor12d
        self.ptr_zobrist_lookup = &zobrist_lookup

        if copy is None:
            self.initialize_new(size, enforce_superko)
        else:
            self.initialize_duplicate(copy)

    ############################################################################
    #   private cdef functions used for game-play                              #
    #                                                                          #
    ############################################################################

    cdef bool is_positional_superko(self, location_t location):
        """Check whether the current player playing at 'location' would result in a state identical
           to a previously seen state. Move must otherwise be legal.
        """

        cdef int i, first
        cdef bool played

        # Part 1: quickly check that the current player has ever played at this location; if not, no
        # fancier check is needed.

        # Check if 'location' is one of the handicap stones placed by BLACK
        if self.current_player == stone_t.BLACK and self.num_handicap > 0:
            played = location in self.moves_history[:self.num_handicap]

        # Calculate which was the first non-handicap move made by the current player
        first = self.num_handicap + (1 if self.current_player == stone_t.WHITE else 0)

        # Check if 'location' matches any other move made by the current player.
        played = played or location in self.moves_history[first::2]

        # If player never played at 'location', superko is impossible and we're done.
        if not played:
            return False

        # Part 2: try move on a duplicate board and check hash of the result.

        # TODO (?) faster hash check than duplicate full object
        # Duplicate state and play move
        cdef GameState copy_state = GameState(copy=self)

        # Do move with superko check disabled (note: move must otherwise be legal).
        copy_state.enforce_superko = False
        copy_state.add_stone(location)

        # Check if hash already exists (hash collisions are very unlikely)
        if copy_state.zobrist_current in self.previous_hashes:
            return True

        return False

    cdef bool has_liberty_after(self, location_t location):
        """Check if a play at location results in an alive group. This is true if
           - there is already a liberty next to this location, or
           - connects to group with >= 2 liberties, or
           - captures an enemy group
        """

        cdef int i
        cdef stone_t board_value
        cdef short count_liberty
        cdef location_t neighbor_loc

        # Check all four neighbors
        for i in range(4):
            # Get neighbor location
            neighbor_loc = d(self.ptr_neighbor)[location * 4 + i]
            board_value = d(self.board[neighbor_loc]).color

            # If neighbor is empty, we're done
            if board_value == stone_t.EMPTY:
                return True

            # Check neighboring group.. this location will have a liberty either if (1) the
            # neighboring group is friendly and has extra liberties, or (2) the neighboring group is
            # the opponent and is captured by this move.
            count_liberty = d(self.board[neighbor_loc]).count_liberty

            # Case (1): neighboring group is friendly. Since playing at location would remove one
            # liberty, this group would need >= 2 liberties to still have one left after playing
            # here.
            if board_value == self.current_player and count_liberty >= 2:
                return True

            # Case (2): neighboring group is opponent. If the opposing group has exactly 1 liberty,
            # it must be at 'location', hence playing here would capture the opposing group.
            elif board_value == self.opponent_player and count_liberty == 1:
                return True

        return False

    cdef void update_legal_moves(self):
        """Update legal_moves list
        """

        cdef location_t loc

        # Clear previous values in self.legal_moves
        self.legal_moves.clear()

        # Loop over all board locations and check if a move is legal.
        # TODO (?) store smaller list of empty locations and search only those
        for loc in range(self.board_size):
            if self.is_legal_move(loc):
                self.legal_moves.push_back(loc)

    cdef void swap_players(self):
        """Switch current_player and opponent_player
        """

        cdef stone_t swap = self.current_player
        self.current_player = self.opponent_player
        self.opponent_player = swap

    cpdef void set_current_player(self, stone_t color):
        """Set current player to the given color.
        """
        if color <= stone_t.EMPTY:
            raise ValueError("Player color must be BLACK or WHITE")
        elif color != self.current_player:
            self.swap_players()
            self.ko = -1
            self.update_legal_moves()

    ############################################################################
    #   private cdef helper functions for feature generation                   #
    #                                                                          #
    ############################################################################

    cdef bool is_eyeish(self, location_t location, stone_t owner):
        """Check if a location is 'eyeish'; that is, check that all 4 neighbors are either border
           or the same color.
        """

        cdef group_ptr_t group
        cdef int i

        # First, 'location' must be empty
        if d(self.board[location]).color != stone_t.EMPTY:
            return False

        # Second, all 4 neighbors must either be BORDER or the same color as owner.
        for i in range(4):
            group = self.board[d(self.ptr_neighbor)[4 * location + i]]
            if not (d(group).color == stone_t.BORDER or d(group).color == owner):
                return False
        return True

    cdef bool is_true_eye(self, location_t location, stone_t owner, list stack=[]):
        """Check if location is a 'real' eye; this goes beyond checking if a location is 'eyeish' by
           checking that corners have the same owner or are themselves eyes, recursively. A group
           with two "true eyes" cannot be captured.
        """

        cdef int i
        cdef stone_t board_value
        cdef short max_bad_diagonal, count_bad_diagonal = 0
        cdef location_t neighbor_loc

        # First, check that location is at least 'eyeish'
        if not self.is_eyeish(location, owner):
            return False

        # If there is any adjacent border, max 'bad' diagonals is 0, otherwise it is 1.
        for i in range(4):
            neighbor_loc = d(self.ptr_neighbor3x3)[location * 8 + i]
            if d(self.board[neighbor_loc]).color == stone_t.BORDER:
                max_bad_diagonal = 0
                break

        # Note: 'else' on a for loop is invoked if there was no break. Here, this means there was no
        # adjacent border.
        else:
            max_bad_diagonal = 1

        # Check diagonal neighbors; they are 'bad' if occupied by an opponent or if empty and not
        # itself an eye (checked recursively)
        for i in range(4, 8):
            neighbor_loc = d(self.ptr_neighbor3x3)[location * 8 + i]
            board_value = d(self.board[neighbor_loc]).color

            # Check if diagonal is enemy stone
            if board_value > stone_t.EMPTY and board_value != owner:
                count_bad_diagonal += 1

            # Check if diagonal is empty and is itself an eye. First check if neighbor is in the
            # search stack, in which case we don't recurse, since that would be an infinite loop.
            elif board_value == stone_t.EMPTY and neighbor_loc not in stack:
                stack.append(location)
                if not self.is_true_eye(neighbor_loc, owner, stack):
                    count_bad_diagonal += 1
                stack.pop()

            # Terminate search if at any point there are more "bad" diagonals than the max allowable
            # for this location.
            if count_bad_diagonal > max_bad_diagonal:
                return False

        # If made it to here, location must be a eye
        return True

    ############################################################################
    #   public cdef functions for feature generation (used by preprocessing)   #
    #   TODO: move all of these to preprocessing itself                        #
    ############################################################################

    cdef pattern_hash_t get_12d_hash(self, location_t center, bool include_player, int max_liberty=3):  # noqa:E501
        """Get unique-ish hash of the 12-stone pattern centered at 'center'. Assumes 'center'
           itself is EMPTY. If 'include_player' is True, hash also takes into account who is the
           current player.
        """

        # First compute hash for all stones around 'center'
        cdef pattern_hash_t hsh = \
            get_pattern_hash(self.board, center, 12, d(self.ptr_neighbor12d), max_liberty)

        # If specified, also include the current player as part of the hash
        if include_player:
            hsh += self.current_player
            hsh *= _HASHVALUE

        return hsh

    cdef pattern_hash_t get_3x3_hash(self, short center, bool include_player, int max_liberty=3):
        """Get unique-ish hash of the 8-stone pattern centered at 'center'. Assumes 'center' itself
           is EMPTY. If 'include_player' is True, hash also takes into account who is the current
           player.
        """
        cdef pattern_hash_t hsh = \
            get_pattern_hash(self.board, center, 8, d(self.ptr_neighbor3x3), max_liberty)

        # If specified, also include the current player as part of the hash
        if include_player:
            hsh += self.current_player
            hsh *= _HASHVALUE

        return hsh

    cdef vector[location_t] get_sensible_moves(self):
        """'Sensible' moves are all legal moves that are not eyes of the current player.
        """

        cdef location_t loc
        cdef vector[location_t] sensible_moves = vector[location_t]()
        cdef list eyes = []  # Keep track of all eyes found so far to speed up search

        for loc in self.legal_moves:
            if self.is_true_eye(loc, self.current_player, eyes):
                eyes.append(loc)
            else:
                sensible_moves.push_back(loc)

        return sensible_moves

    ############################################################################
    #   public cdef functions used for game play                               #
    #                                                                          #
    ############################################################################

    cpdef bool is_legal_move(self, location_t location):
        """Check if playing at location is a legal move
        """

        # Passing is always legal
        if location == action_t.PASS:
            return True

        # Check that location is on the board
        if location < 0 or location >= self.board_size:
            return False

        # Check if it is empty
        if d(self.board[location]).color > stone_t.EMPTY:
            return False

        # Check ko (1 move back only)
        if location == self.ko:
            return False

        # Check if move is suicide
        if not self.has_liberty_after(location):
            return False

        # (Maybe) check superko state
        if self.enforce_superko and self.is_positional_superko(location):
            return False

        # If all of the above checks pass, then this move is legal.
        return True

    cdef location_t add_stone(self, location_t location):
        """Play stone on location and update the state. MOVE MUST BE LEGAL. Returns new ko location,
           or -1 if there is none.
        """

        cdef group_ptr_t captured_stones = group_new(self.opponent_player)
        cdef group_ptr_t new_group = group_new(self.current_player)
        cdef group_ptr_t neighbor_group
        cdef stone_t neighbor
        cdef location_t neighbor_loc, new_ko = -1
        cdef int i

        # Start new group for this stone.
        group_add_stone(new_group, location)

        # Add new group to groups_set and ensure board location is updated.
        self.groups_set.insert(new_group)
        self.board[location] = new_group

        # Check neighbors: merge friendly groups and (maybe) capture opponents
        for i in range(4):
            neighbor_loc = d(self.ptr_neighbor)[location * 4 + i]
            neighbor_group = self.board[neighbor_loc]
            neighbor = d(neighbor_group).color

            # Add liberties to the new group
            if neighbor == stone_t.EMPTY:
                group_add_liberty(new_group, neighbor_loc)

            # Merge neighboring friendly groups (and remove 'location' as one of their liberties)
            elif neighbor == self.current_player and neighbor_group != new_group:
                group_remove_liberty(neighbor_group, location)
                # Note: order matters here; calling combine_groups(neighbor_group, new_group) would
                # interfere with TemporaryMove, which does nothing to 'un-combine' the new stone
                # from 'neighbor_group'. As written here, stones are copied from 'neighbor_group'
                # into 'new_group', but 'neighbor_group' itself is left unchanged.
                new_group = self.combine_groups(new_group, neighbor_group)

            # Handle stone placed next to opponent group: remove stone from opponent liberties and
            # check for capture.
            elif neighbor == self.opponent_player:
                group_remove_liberty(neighbor_group, location)

                # Capture opponent if this stone covered opponent's last liberty.
                if d(neighbor_group).count_liberty == 0:
                    group_merge(captured_stones, neighbor_group)
                    self.remove_group(neighbor_group)

        # Count captured stones
        if self.current_player == stone_t.BLACK:
            self.capture_white += d(captured_stones).count_stones
        else:
            self.capture_black += d(captured_stones).count_stones

        # Update ko: ko occurs when both captured group and newly created group are size 1.
        if d(captured_stones).count_stones == 1 and d(new_group).count_stones == 1:
            new_ko = group_get_stone(captured_stones)

        # Update zobrist hash: first, update with newly added stone
        self.zobrist_current = update_hash_by_location(self.zobrist_current,
                                                       d(self.ptr_zobrist_lookup),
                                                       location, self.current_player)
        # Second, update with all captured stones.
        if d(captured_stones).count_stones > 0:
            self.zobrist_current = update_hash_by_group(self.zobrist_current,
                                                        d(self.ptr_zobrist_lookup),
                                                        captured_stones)
        self.previous_hashes.add(self.zobrist_current)

        return new_ko

    cpdef TemporaryMove try_stone(self, location_t location, bool prepare_next=True):
        """Analogous to add_stone() for use in a with-statement. Automatically undoes the given move
           when the with-statement exits.

           For example:

               state.add_stone(loc1)
               with state.try_stone(loc2):
                   print state.get_history()[-1]  # prints loc2
               # Here, state is returned to its value to before the with statement.
               print state.get_history()[-1]  # prints loc1
        """

        return TemporaryMove(self, location, prepare_next)

    cpdef list get_legal_moves(self, bool include_eyes=True):
        """Return a list with all legal moves as tuples (in/excluding eyes)
        """

        cdef list moves

        if include_eyes:
            moves = list(self.legal_moves)
        else:
            moves = list(self.get_sensible_moves())

        return [calculate_tuple_location(m, self.size) for m in moves]

    cpdef float get_score(self, float komi=7.5):
        """Calculate score of board state. Uses 'Area scoring'.

           http://senseis.xmp.net/?Passing#1

           Negative value indicates black win, positive value indicates white win.
        """

        cdef location_t location
        cdef stone_t board_value

        # Positive score is in favor of black, negative is in favor of white.
        cdef float score = -komi

        # Keep track of all eyes for both black and white to make search faster.
        cdef list eyes_white = [], eyes_black = []

        # Loop over whole board
        for location in range(self.board_size):
            board_value = d(self.board[location]).color

            # Decrement score difference for white
            if board_value == stone_t.WHITE:
                score -= 1

            # Increment score difference for black
            elif board_value == stone_t.BLACK:
                score += 1

            # If empty, count as territory only if it is an eye
            else:
                if self.is_true_eye(location, stone_t.BLACK, eyes_black):
                    eyes_black.append(location)
                    score += 1

                elif self.is_true_eye(location, stone_t.WHITE, eyes_white):
                    eyes_white.append(location)
                    score -= 1

        # substract passes
        # http://senseis.xmp.net/?Passing#1
        score -= self.passes_black
        score += self.passes_white

        return score

    cpdef stone_t get_winner_color(self, float komi=7.5):
        """Calculate score of board state and return winning player (WHITE or BLACK). Uses 'Area
           scoring'. Tie goes to WHITE.
        """

        cdef float score = self.get_score(komi)

        # BLACK wins for strictly positive score. 0 (tie) or negative goes to WHITE.
        if score > 0:
            return stone_t.BLACK
        else:
            return stone_t.WHITE

    cpdef void do_move(self, tuple action, stone_t color=stone_t.EMPTY):
        """Play stone at action=(x,y). Use action=None for passing. Checks move legality first.

           If it is a legal move, current_player switches to the opposite color. If not, an
           IllegalMove exception is raised
        """

        cdef location_t x, y, location
        cdef group_ptr_t grp

        # Note: as per the python interface, 'None' is considerd a pass
        if action is None:
            location = action_t.PASS

            if self.current_player == stone_t.BLACK:
                self.passes_black += 1
            else:
                self.passes_white += 1

            # Reset ko since players switched.
            self.ko = -1

        else:
            if color != stone_t.EMPTY and color != self.current_player:
                self.swap_players()

            # Convert from tuple (x, y) input to 1d coordinate.
            (x, y) = action
            location = calculate_board_location(y, x, self.size)

            # Check if move is legal.
            if not self.is_legal_move(location):
                raise IllegalMove(str(action))

            # Execute move.
            self.ko = self.add_stone(location)

        # Add move to history
        self.moves_history.push_back(location)

        # Swap current player for next turn.
        self.swap_players()

        # The set of legal moves must now be recomputed, since it is different for each player
        # and with new ko location.
        self.update_legal_moves()

    cpdef void place_handicap_stone(self, tuple action, stone_t color=stone_t.BLACK):
        """Add handicap stones given by a list of tuples in list handicap
        """

        if self.moves_history.size() > self.num_handicap:
            raise IllegalMove("Cannot place handicap on a started game")

        self.num_handicap += 1
        self.do_move(action, color)

    cpdef void place_handicaps(self, list handicap):
        """Place list of handicap stones for BLACK (must be on empty board)
        """

        for action in handicap:
            self.place_handicap_stone(action, stone_t.BLACK)

    ############################################################################
    #   Helper functions for managing groups                                   #
    #                                                                          #
    ############################################################################

    cdef group_ptr_t combine_groups(self, group_ptr_t group_keep, group_ptr_t group_remove):
        """Combine group_keep and group_remove by copying all stones and liberties from
           group_remove into group_keep, and update pointers on the board. Leaves group_remove
           unchanged, but does remove it from groups_set.

           Returns group_keep
        """

        cdef location_t loc
        cdef group_t val

        # Loop over all location->stone pairs in group_remove.
        for loc, val in d(group_remove).locations:
            if val == group_t.STONE:
                # group_remove has a stone at this location; add the stone to group_keep and set
                # board location to group_keep.
                group_add_stone(group_keep, loc)
                self.board[loc] = group_keep
            elif val == group_t.LIBERTY:
                # group_remove has a liberty at this location; add the liberty to group_keep.
                group_add_liberty(group_keep, loc)

        # Clear group_remove from groups_set.
        self.groups_set.erase(group_remove)

        return group_keep

    cdef void remove_group(self, group_ptr_t group_remove):
        """Remove group from everywhere in the state.
        """

        cdef location_t loc, neighbor_loc
        cdef group_ptr_t group
        cdef group_t val, neighbor_value
        cdef int i

        # First pass: clear stones by setting board[loc] to EMPTY for each stone in this group.
        for loc, val in d(group_remove).locations:
            if val == group_t.STONE:
                # Set board at this location to empty group
                self.board[loc] = self.group_empty

        # Second pass: add new liberties to neighboring groups now that stones have been cleared.
        for loc, val in d(group_remove).locations:
            if val == group_t.STONE:
                for i in range(4):
                    # Get neighbor location
                    neighbor_loc = d(self.ptr_neighbor)[loc * 4 + i]
                    group = self.board[neighbor_loc]

                    # If neighbor is a stone, add 'loc' as a liberty
                    if d(group).color > stone_t.EMPTY:
                        group_add_liberty(group, loc)

        # Clear this group from groups_set
        self.groups_set.erase(group_remove)

    ############################################################################
    #   Python convenience functions (not declared in .pxd)                    #
    #                                                                          #
    ############################################################################

    def is_end_of_game(self):
        if self.moves_history.size() > 1:
            if self.moves_history[self.moves_history.size() - 1] == action_t.PASS and \
                    self.moves_history[self.moves_history.size() - 2] == action_t.PASS and \
                    self.current_player == stone_t.WHITE:
                return True
        return False

    def is_legal(self, action):
        """Determine if the given action (x,y tuple) is a legal move
        """

        cdef char x, y
        cdef short location
        (x, y) = action

        # Check outside board
        if x < 0 or y < 0 or x >= self.size or y >= self.size:
            return False

        # Calculate 1D location
        location = calculate_board_location(y, x, self.size)

        return self.is_legal_move(location)

    def copy(self):
        """Get a copy of this Game state
        """

        return GameState(copy=self)

    ############################################################################
    #   public def functions used for unittests                                #
    #                                                                          #
    ############################################################################

    cpdef void print_groups(self):
        """Debugging helper: prints address of all groups in this state and info about each one.
        """

        cdef location_t loc
        cdef group_t val
        cdef group_ptr_t group

        print("Empty: {0:x}".format(<unsigned long long> self.group_empty.get()))
        print("Border: {0:x}".format(<unsigned long long> self.group_border.get()))

        for loc in range(self.board_size):
            if self.board[loc].get() != self.group_empty.get():
                print("%03d: %x" % (loc, <unsigned long long> self.board[loc].get()))

        for group in self.groups_set:
            print("--- %x: %d / %d / %d ---" % (<unsigned long long> group.get(), d(group).color,
                                                d(group).count_stones, d(group).count_liberty))
            for loc, val in d(group).locations:
                print "\t", calculate_tuple_location(loc, self.size), val

    cpdef bool sanity_check_groups(self):
        """Debugging helper: loops over every location and group on the board and checks that they are
           self-consistent.
        """

        cdef location_t loc, neighbor_loc
        cdef group_ptr_t group1, group2, neighbor_group
        cdef group_t val
        cdef int i, recount_liberty, recount_stones
        cdef set empty_adjacent

        # Check 1: all groups in self.board are also in self.groups_set.
        for loc in range(self.board_size):
            group1 = self.board[loc]

            # Skip empty group.
            if group1 == self.group_empty:
                continue

            # Check for this group in self.groups_set.
            for group2 in self.groups_set:
                if group2 == group1:
                    break
            else:
                print("board[loc] points to nonexistent group!")
                return False

        # Check 2: vice versa
        for group1 in self.groups_set:
            for loc in range(self.board_size):
                if group1 == self.board[loc]:
                    break
            else:
                print("groups_set contains a group not on the board!")
                return False

        # Check 3: all neighbors of the same color are in the same group.
        for loc in range(self.board_size):
            group1 = self.board[loc]

            # Only check locations with stones.
            if d(group1).color > stone_t.EMPTY:
                for i in range(4):
                    neighbor_loc = d(self.ptr_neighbor)[4 * loc + i]
                    neighbor_group = self.board[neighbor_loc]

                    # If neighbor has stone of the same color, they must be the same group object.
                    if d(neighbor_group).color == d(group1).color:
                        if neighbor_group != group1:
                            print("neighbors of same color not in the same group!")
                            return False

        # Check 4: group's counts and locations are consistent with the board.
        for group1 in self.groups_set:
            recount_liberty, recount_stones = 0, 0
            empty_adjacent = set()
            for loc, val in d(group1).locations:
                # Check that group's STONE locations are correct in self.board
                if val == group_t.STONE:
                    if self.board[loc] != group1:
                        print("group has STONE but board does not point to that group!")
                        return False
                    recount_stones += 1

                    # Check for neighboring liberties on the board
                    for i in range(4):
                        neighbor_loc = d(self.ptr_neighbor)[4 * loc + i]
                        if self.board[neighbor_loc] == self.group_empty:
                            empty_adjacent.add(neighbor_loc)

                # Check that group's LIBERTY locations are actually empty
                elif val == group_t.LIBERTY:
                    if self.board[loc] != self.group_empty:
                        print("group has LIBERTY but board does not point to empty group!")
                        return False
                    recount_liberty += 1

            # Check that all counts make sense
            if recount_stones != d(group1).count_stones:
                print("mismatch in stones count!")
                return False

            elif recount_liberty != d(group1).count_liberty:
                print("mismatch in liberties count!")
                return False

            elif recount_liberty != len(empty_adjacent):
                print("liberty count does not match actuall number of empty adjacent locations!")
                return False

        # All checks passed
        return True

    def get_current_player(self):
        """Returns the color of the player who will make the next move.
        """

        return self.current_player

    def get_history(self):
        """Return history as a list of tuples
        """

        return [calculate_tuple_location(loc, self.size) for loc in self.moves_history]

    def get_captures_black(self):
        """Return amount of black stones captures
        """

        return self.capture_black

    def get_captures_white(self):
        """Return amount of white stones captured
        """

        return self.capture_white

    def get_ko_location(self):
        """Return ko location as a tuple, or None
        """

        if self.ko == -1:
            return None

        return calculate_tuple_location(self.ko, self.size)

    def is_board_equal(self, GameState state):
        """Verify that self and state board layout are the same
        """

        for i in range(self.board_size):
            if d(self.board[i]).color != d(state.board[i]).color:
                return False

        return True

    def is_liberty_equal(self, GameState state):
        """Verify that self and state liberty counts are the same
        """

        for i in range(self.board_size):
            if d(self.board[i]).count_liberty != d(state.board[i]).count_liberty:
                return False

        return True

    def is_eye(self, action, color):
        """Check if location action is a eye for player color
        """

        (x, y) = action
        location = calculate_board_location(y, x, self.size)

        return self.is_true_eye(location, color)

    def get_liberty(self):
        """Get numpy array with all liberty counts for all stones.
        """

        liberty = np.zeros((self.size, self.size), dtype=np.int)

        for x in range(self.size):
            for y in range(self.size):
                location = calculate_board_location(y, x, self.size)
                liberty[x, y] = d(self.board[location]).count_liberty

        return liberty

    def get_board(self):
        """Get numpy array with board locations set to stone colors
        """

        board = np.zeros((self.size, self.size), dtype=np.int)

        for x in range(self.size):
            for y in range(self.size):
                location = calculate_board_location(y, x, self.size)
                board[x, y] = d(self.board[location]).color

        return board

    def get_hash(self):
        return self.zobrist_current

    def get_size(self):
        """Return size
        """

        return self.size

    def get_handicaps(self):
        """Return list with handicap stones placed by BLACK at beginning of the game.
        """

        return self.moves_history[:self.num_handicap]

    def get_print_board_layout(self):
        """Print current board state
        """

        line = "\n"
        for i in range(self.size):
            row = str(i) + " "
            for j in range(self.size):
                stone = '.'
                if d(self.board[j + i * self.size]).color == stone_t.BLACK:
                    stone = 'B'
                elif d(self.board[j + i * self.size]).color == stone_t.WHITE:
                    stone = 'W'
                row += stone + " "
            line += row + "\n"
        return line

    def __repr__(self):
        """Enable python: print GameState
        """

        return self.get_print_board_layout()

    def __str__(self):
        """Enable python: str(GameState)
        """

        return self.get_print_board_layout()


cdef class TemporaryMove:
    """Helper-class for GameState.try_stone(). Must be called with a legal move.

       This class implements python's 'with' interface such that when the 'with' block is exited,
       the move is undone.

       See https://www.python.org/dev/peps/pep-0343/ for more information about how 'with' interacts
       with __enter__ and __exit__
    """

    def __init__(self, GameState state, location_t move, bool prepare_next):
        self.state = state
        self.move = move
        self.neighbors_friendly = group_set_t()
        self.neighbors_opponent = group_set_t()
        self.prepare_next = prepare_next

    def __enter__(self):
        """Called when entering 'with' statement. Execute the given move and record necessary
           information about the state so that 'move' may be undone later.
        """

        cdef location_t neighbor_loc
        cdef group_ptr_t neighbor_group
        cdef int i

        # Record player color and ko.
        self.player_color = self.state.current_player
        self.previous_ko = self.state.ko

        # Get a reference to each of the up-to-4 neighbors of the stone about to be placed so they
        # can be restored in __exit__.
        for i in range(4):
            neighbor_loc = d(self.state.ptr_neighbor)[self.move * 4 + i]
            neighbor_group = self.state.board[neighbor_loc]
            if d(neighbor_group).color == self.player_color:
                self.neighbors_friendly.insert(neighbor_group)
            elif d(neighbor_group).color > stone_t.EMPTY:
                self.neighbors_opponent.insert(neighbor_group)

        if self.prepare_next:
            # Call the same state-updating methods as do_move.
            self.state.ko = self.state.add_stone(self.move)
            self.state.moves_history.push_back(self.move)
            self.state.swap_players()
            self.state.update_legal_moves()
        else:
            # Only add the stone and don't perform further updates.
            self.state.add_stone(self.move)

        return self.state

    def __exit__(self, type, value, traceback):
        """Called at end of 'with' statement. Undo move done in __enter__.
        """

        cdef group_ptr_t old_group, neighbor_group
        cdef location_t loc, neighbor_loc
        cdef group_t val
        cdef int i, c

        # Take away current hash from set of hashes.
        self.state.previous_hashes.discard(self.state.zobrist_current)

        # Update hash: remove placed stone.
        self.state.zobrist_current = update_hash_by_location(self.state.zobrist_current,
                                                             d(self.state.ptr_zobrist_lookup),
                                                             self.move, self.player_color)

        # Remove group that the new stone belongs to and set its board location to empty.
        self.state.groups_set.erase(self.state.board[self.move])
        self.state.board[self.move] = self.state.group_empty

        # Remove 'new' neighbor groups to prepare to restore 'old' neighbor groups. Update to
        # state.board[loc] for neighbors happens below.
        for i in range(4):
            neighbor_loc = d(self.state.ptr_neighbor)[self.move * 4 + i]
            neighbor_group = self.state.board[neighbor_loc]
            if d(neighbor_group).color > stone_t.EMPTY:
                self.state.groups_set.erase(neighbor_group)

        # Restore previous FRIENDLY neighbors. It is important that this happens before restoring
        # opponent so that if a captured opponent group is restored, it updates liberties of the
        # correct friendly group.
        for old_group in self.neighbors_friendly:
            # Ensure group is in 'groups_set'.
            self.state.groups_set.insert(old_group)

            # Point board[loc] to this group for each stone in the group.
            for loc, val in d(old_group).locations:
                if val == group_t.STONE:
                    self.state.board[loc] = old_group

            # Restore liberty of old group.
            group_add_liberty(old_group, self.move)

        # Restore previous OPPONENT neighbors.
        for old_group in self.neighbors_opponent:
            # Ensure group is in 'groups_set'.
            self.state.groups_set.insert(old_group)

            # Point board[loc] to this group for each stone in the group.
            for loc, val in d(old_group).locations:
                if val == group_t.STONE:
                    self.state.board[loc] = old_group

            # Restore liberty of old group.
            group_add_liberty(old_group, self.move)

            # Capture had occurred if old_group has 1 liberty after having 'restored' the liberty
            # (in other words, if it had 0 liberties a moment ago).
            if d(old_group).count_liberty == 1:
                # Restore hash for each captured stone.
                self.state.zobrist_current = \
                    update_hash_by_group(self.state.zobrist_current,
                                         d(self.state.ptr_zobrist_lookup), old_group)

                # Undo 'remove_group' by replacing stones and removing liberties of other
                # surrounding groups.
                for loc, val in d(old_group).locations:
                    if val == group_t.STONE:
                        # 'loc' is being re-added to the board; must update neighbor liberties.
                        for i in range(4):
                            neighbor_loc = d(self.state.ptr_neighbor)[loc * 4 + i]
                            neighbor_group = self.state.board[neighbor_loc]

                            # Remove liberty of neighbor group if it has stones.
                            if d(neighbor_group).color > stone_t.EMPTY:
                                group_remove_liberty(neighbor_group, loc)

                # Decrement captured stones.
                if self.player_color == stone_t.BLACK:
                    self.state.capture_white -= d(old_group).count_stones
                else:
                    self.state.capture_black -= d(old_group).count_stones

        if self.prepare_next:
            # Remove stone from history.
            self.state.moves_history.pop_back()

            # Restore ko.
            self.state.ko = self.previous_ko

            # Switch back to other player.
            self.state.swap_players()

            # Restore set of legal moves
            self.state.update_legal_moves()


class IllegalMove(Exception):
    pass
