cimport cython
from libc.stdlib cimport malloc, free

cdef class RootState:

    ############################################################################
    #   variables declarations                                                 #
    #                                                                          #
    ############################################################################

    """ -> variables, declared in go_root.pxd

    cdef short  size

    # arrays, neighbor arrays pointers
    cdef short *neighbor
    cdef short *neighbor3x3
    cdef short *neighbor12d

    """

    ############################################################################
    #   cdef init functions                                                    #
    #                                                                          #
    ############################################################################


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef short calculate_board_location( self, char x, char y ):
        """
           return location on board
           no checks on outside board
           x = columns
           y = rows           
        """

        # return board location
        return x + ( y * self.size )


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef short calculate_board_location_or_border( self, char x, char y ):
        """
           return location on board or borderlocation
           board locations = [ 0, size * size )
           border location = size * size
           x = columns
           y = rows
        """

        # check if x or y are outside board
        if x < 0 or y < 0 or x >= self.size or y >= self.size:
            # return border location
            return self.board_size

        # return board location
        return self.calculate_board_location( x, y )


    cdef void set_neighbors( self, int size ):
        """
           create array for every board location with all 4 direct neighbour locations
           neighbor order: left - right - above - below

                    -1     x 
                          x x
                    +1     x 

                    order:
                    -1     2 
                          0 1
                    +1     3 

           TODO neighbors is obsolete as neighbor3x3 contains the same values 
        """

        # create array
        self.neighbor = <short *>malloc( size * size * 4 * sizeof( short ) )
        if not self.neighbor:
            raise MemoryError()

        cdef short location
        cdef char x, y

        # add all direct neighbors to every board location
        for y in range( size ):
            for x in range( size ):
                location = ( x + ( y * size ) ) * 4
                self.neighbor[ location + 0 ] = self.calculate_board_location_or_border( x - 1, y     )
                self.neighbor[ location + 1 ] = self.calculate_board_location_or_border( x + 1, y     )
                self.neighbor[ location + 2 ] = self.calculate_board_location_or_border( x    , y - 1 )
                self.neighbor[ location + 3 ] = self.calculate_board_location_or_border( x    , y + 1 )


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef void set_3x3_neighbors( self, int size ):
        """
           create for every board location array with all 8 surrounding neighbour locations
           neighbor order: above middle - middle left - middle right - below middle
                           above left - above right - below left - below right
                           this order is more useful as it separates neighbors and then diagonals
                    -1    xxx
                          x x
                    +1    xxx           

                    order:
                    -1    405
                          1 2
                    +1    637           

            0-3 contains neighbors
            4-7 contains diagonals
        """

        # create array
        self.neighbor3x3 = <short *>malloc( size * size * 8 * sizeof( short ) )
        if not self.neighbor3x3:
            raise MemoryError()

        cdef short location
        cdef char x, y

        # add all surrounding neighbors to every board location
        for x in range( size ):
            for y in range( size ):
                location = ( x + ( y * size ) ) * 8
                self.neighbor3x3[ location + 0 ] = self.calculate_board_location_or_border( x    , y - 1 )
                self.neighbor3x3[ location + 1 ] = self.calculate_board_location_or_border( x - 1, y     )
                self.neighbor3x3[ location + 2 ] = self.calculate_board_location_or_border( x + 1, y     )
                self.neighbor3x3[ location + 3 ] = self.calculate_board_location_or_border( x    , y + 1 )

                self.neighbor3x3[ location + 4 ] = self.calculate_board_location_or_border( x - 1, y - 1 )
                self.neighbor3x3[ location + 5 ] = self.calculate_board_location_or_border( x + 1, y - 1 )
                self.neighbor3x3[ location + 6 ] = self.calculate_board_location_or_border( x - 1, y + 1 )
                self.neighbor3x3[ location + 7 ] = self.calculate_board_location_or_border( x + 1, y + 1 )


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef void set_12d_neighbors( self, int size ):
        """
           create array for every board location with 12d star neighbour locations
           neighbor order: top star tip
                           above left - above middle - above right
                           left star tip - left - right - right star tip
                           below left - below middle - below right
                           below star tip
     
                    -2     x 
                    -1    xxx
                         xx xx
                    +1    xxx
                    +2     x        

                    order:
                    -2     0 
                    -1    123
                         45 67
                    +1    89a
                    +2     b    
        """

        # create array
        self.neighbor12d = <short *>malloc( size * size * 12 * sizeof( short ) )
        if not self.neighbor12d:
            raise MemoryError()

        cdef short location
        cdef char x, y

        # add all 12d neighbors to every board location
        for x in range( size ):
            for y in range( size ):
                location = ( x + ( y * size ) ) * 12
                self.neighbor12d[ location +  4 ] = self.calculate_board_location_or_border( x    , y - 2 )

                self.neighbor12d[ location +  1 ] = self.calculate_board_location_or_border( x - 1, y - 1 )
                self.neighbor12d[ location +  5 ] = self.calculate_board_location_or_border( x    , y - 1 )
                self.neighbor12d[ location +  8 ] = self.calculate_board_location_or_border( x + 1, y - 1 )

                self.neighbor12d[ location +  0 ] = self.calculate_board_location_or_border( x - 2, y     )
                self.neighbor12d[ location +  2 ] = self.calculate_board_location_or_border( x - 1, y     )
                self.neighbor12d[ location +  9 ] = self.calculate_board_location_or_border( x + 1, y     )
                self.neighbor12d[ location + 11 ] = self.calculate_board_location_or_border( x + 2, y     )

                self.neighbor12d[ location +  3 ] = self.calculate_board_location_or_border( x - 1, y + 1 )
                self.neighbor12d[ location +  6 ] = self.calculate_board_location_or_border( x    , y + 1 )
                self.neighbor12d[ location + 10 ] = self.calculate_board_location_or_border( x + 1, y + 1 )

                self.neighbor12d[ location +  7 ] = self.calculate_board_location_or_border( x    , y + 2 )


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def __init__( self, char size = 19 ):
        """
           RootState initializes all neighbor arrays
           when destoyed all arrays are freed in order to prevent a memory leak
        """

        # set size
        self.size = size
        self.board_size = size * size

        # initialize neighbor locations
        self.set_neighbors(     size )
        self.set_3x3_neighbors( size )
        self.set_12d_neighbors( size )

        # initialize EMPTY and BORDER group
        self.group_empty  = group_new( _EMPTY,  self.board_size )
        self.group_border = group_new( _BORDER, self.board_size )


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def __dealloc__(self):
        """
           this function is called when this object is destroyed

           Prevent memory leaks by freeing all arrays created with malloc
        """

        # free neighbor
        if self.neighbor is not NULL:
            free( self.neighbor )

        # free neighbor3x3
        if self.neighbor3x3 is not NULL:
            free( self.neighbor3x3 )

        # free neighbor12d
        if self.neighbor12d is not NULL:
            free( self.neighbor12d )

        # free border and empty group
        if self.group_empty is not NULL:

            free( self.group_empty  )

        if self.group_border is not NULL:

            free( self.group_border )


    ############################################################################
    #   public functions ( c only )                                            #
    #                                                                          #
    ############################################################################


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef GameState get_root_state( self ):
        """
           
        """

        cdef GameState game_state = GameState()

        cdef short i

        # set pointer to neighbor locations
        game_state.neighbor    = self.neighbor
        game_state.neighbor3x3 = self.neighbor3x3
        game_state.neighbor12d = self.neighbor12d

        # initialize size and board_size
        game_state.size        = self.size
        game_state.board_size  = self.size * self.size

        # create history list
        game_state.history     = []

        # initialize player colours
        game_state.player_current  = _BLACK
        game_state.player_opponent = _WHITE

        game_state.ko              = _PASS
        game_state.capture_black   = 0
        game_state.capture_white   = 0
        game_state.passes_black    = 0
        game_state.passes_white    = 0

        # create arrays and lists
        # +1 on board_size is used as an border location used for all borders

        # create board_groups array able to hold all groups on the board and border ( board_size +1 )
        game_state.board_groups = <Group **>malloc( ( self.board_size + 1 ) * sizeof( Group* ) )
        if not game_state.board_groups:
            raise MemoryError()

        # create 3x3 hash array
        game_state.hash3x3      = <long   *>malloc( ( self.board_size     ) * sizeof( long   ) )
        if not game_state.hash3x3:
            raise MemoryError()

        # create Locations_List able to hold all legal moves ( == board_size )
        game_state.moves_legal            = <Locations_List *>malloc( sizeof( Locations_List ) )
        if not game_state.moves_legal:
            raise MemoryError()

        game_state.moves_legal.locations  = <short *>malloc( self.board_size * sizeof( short ) )
        if not game_state.moves_legal.locations:
            raise MemoryError()

        game_state.moves_legal.count      = self.board_size

        # create groups_list array to hold all groups on the board ( == board_size )
        # TODO estimate a good lower bound
        game_state.groups_list                   = <Groups_List *>malloc( sizeof( Groups_List ) )
        if not game_state.groups_list:
            raise MemoryError()

        game_state.groups_list.board_groups      = <Group **>malloc( self.board_size * sizeof( Group* ) )
        if not game_state.groups_list.board_groups:
            raise MemoryError()

        game_state.groups_list.count_groups      = 0

        # create empty location group
        game_state.group_empty = self.group_empty

        # initialize board
        for i in range( game_state.board_size ):

            game_state.hash3x3[ i ]      = 0
            game_state.board_groups[ i ] = game_state.group_empty
            game_state.moves_legal.locations[ i ] = i

        # initialize border location
        game_state.board_groups[ self.board_size ] = self.group_border
        
        # initialize zobrist hash
        # TODO optimize?
        # rng = np.random.RandomState(0)
        # self.hash_lookup = {
        #    WHITE: rng.randint(np.iinfo(np.uint64).max, size=(size, size), dtype='uint64'),
        #    BLACK: rng.randint(np.iinfo(np.uint64).max, size=(size, size), dtype='uint64')}
        # self.current_hash = np.uint64(0)
        # self.previous_hashes = set()

        return game_state

    ############################################################################
    #   public functions                                                       #
    #                                                                          #
    ############################################################################

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def get_root_game_state( self, enforce_superko=False ):
        """
           return new GameState initialized as a new game
        """

        return self.get_root_state()