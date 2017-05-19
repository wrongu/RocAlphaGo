#cython: profile=True
#cython: linetrace=True
import sys
import time
cimport cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
from datetime import datetime
from datetime import timedelta

# global
# global empty group
cdef Group *group_empty

# global border group
cdef Group *group_border

# global arrays, neighbor arrays pointers
cdef short *neighbor
cdef short *neighbor3x3
cdef short *neighbor12d
cdef char  neighbor_size

# expose variables to python
PASS  = _PASS
BLACK = _BLACK
WHITE = _WHITE
EMPTY = _EMPTY

cdef class GameState:

    ############################################################################
    #   all variables are declared in the .pxd file                            #
    #                                                                          #
    ############################################################################

    """ -> variables, declared in go.pxd

    # amount of locations on one side
    cdef char  size
    # amount of locations on board, size * size
    cdef short board_size

    # possible ko location
    cdef short ko                

    # list with all groups
    cdef Groups_List *groups_list
    # pointer to empty group
    cdef Group       *group_empty

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

    # array, keep track of 3x3 pattern hashes
    cdef long  *hash3x3

    # arrays, neighbor arrays pointers
    cdef short *neighbor
    cdef short *neighbor3x3
    cdef short *neighbor12d

    # zobrist
    cdef dict   hash_lookup
    cdef int    current_hash
    cdef set    previous_hashes

        -> variables, declared in go.pxd
    """

    ############################################################################
    #   init functions                                                         #
    #                                                                          #
    ############################################################################


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef void initialize_new( self, char size ):
        """
           initialize this state as empty state
        """

        cdef short i

        # set pointer to neighbor locations
        # neighbor, neighbor3x3, neighbor12d are global
        self.neighbor    = neighbor
        self.neighbor3x3 = neighbor3x3
        self.neighbor12d = neighbor12d

        # initialize size and board_size
        self.size        = size
        self.board_size  = size * size

        # create history list
        self.moves_history = locations_list_new( 10 )

        # initialize player colours
        self.player_current  = _BLACK
        self.player_opponent = _WHITE

        self.ko              = _PASS
        self.capture_black   = 0
        self.capture_white   = 0
        self.passes_black    = 0
        self.passes_white    = 0

        # create arrays and lists
        # +1 on board_size is used as an border location used for all borders

        # create Group pointer array ( Group **)
        # this array represent the board, every group contains colour, stone-locations
        # and liberty locations
        # border location is included, therefore the array size is board_size +1
        self.board_groups = <Group **>malloc( ( self.board_size + 1 ) * sizeof( Group* ) )
        if not self.board_groups:
            raise MemoryError()

        # create 3x3 hash array, these are updated after every move
        self.hash3x3      = <long   *>malloc( ( self.board_size     ) * sizeof( long   ) )
        if not self.hash3x3:
            raise MemoryError()

        # create Locations_List as legal_moves
        # after every move this list will be updated to contain all legal moves
        # max amount of legal moves is board_size
        self.moves_legal       = locations_list_new( self.board_size )

        # create groups_list as groups_list
        # this list will contain all alive groups
        # we do not need to set the theoretical max amount of groups as the list 
        # will be incremented in group_list_add
        self.groups_list = groups_list_new( self.board_size )

        # get global group_empty reference -> used when removing groups
        self.group_empty = group_empty

        # initialize board, set all locations to group empty and add all
        # locations as move_legal
        for i in range( self.board_size ):

            self.board_groups[ i ]          = group_empty
            self.moves_legal.locations[ i ] = i

        # on an empty board board_size == amount of legal moves
        # set the moves_legal count to board_size
        self.moves_legal.count = self.board_size

        # initialize border location to group_border
        self.board_groups[ self.board_size ] = group_border

        # initialize all 3x3 hashes
        for i in range( self.board_size ):

            self.hash3x3[ i ] = self.generate_3x3_hash( i )
        
        # initialize zobrist hash
        # TODO optimize?
        # rng = np.random.RandomState(0)
        # self.hash_lookup = {
        #    WHITE: rng.randint(np.iinfo(np.uint64).max, size=(size, size), dtype='uint64'),
        #    BLACK: rng.randint(np.iinfo(np.uint64).max, size=(size, size), dtype='uint64')}
        # self.current_hash = np.uint64(0)
        # self.previous_hashes = set()


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef void initialize_duplicate( self, GameState copy_state ):
        """
           Initialize all variables as a copy of copy_state
        """

        cdef int    i
        cdef short  location
        cdef Group* group_pointer
        cdef Group* group

        # !!! do not copy !!! 
        # these do not need a deep copy as they are static
        self.neighbor        = copy_state.neighbor
        self.neighbor3x3     = copy_state.neighbor3x3
        self.neighbor12d     = copy_state.neighbor12d

        # empty group
        self.group_empty     = copy_state.group_empty

        # pattern dictionary

        # zobrist
      # self.hash_lookup     = copy_state.hash_lookup

        # !!! deep copy !!!

        # set all values
        self.ko              = copy_state.ko
        self.capture_black   = copy_state.capture_black
        self.capture_white   = copy_state.capture_white
        self.passes_black    = copy_state.passes_black
        self.passes_white    = copy_state.passes_white
        self.size            = copy_state.size
        self.board_size      = copy_state.board_size
        self.player_current  = copy_state.player_current
        self.player_opponent = copy_state.player_opponent
      # self.current_hash    = copy_state.current_hash

        # create history list
        self.moves_history       = locations_list_new( copy_state.moves_history.size )
        self.moves_history.count = copy_state.moves_history.count
        # copy all history moves in copy_state
        memcpy( self.moves_history.locations, copy_state.moves_history.locations, copy_state.moves_history.count * sizeof( short ) )

      # self.previous_hashes = list( copy_state.previous_hashes )

        # create 3x3 hash array, these are updated after every move
        self.hash3x3         = <long *>malloc( ( self.board_size     ) * sizeof( long ) )
        if not self.hash3x3:
            raise MemoryError()
        # copy all 3x3 hashes from copy_state
        memcpy( self.hash3x3, copy_state.hash3x3, ( self.board_size  ) * sizeof( long ) )

        # create Locations_List as legal_moves
        # after every move this list will be updated to contain all legal moves
        # max amount of legal moves is board_size
        self.moves_legal       = locations_list_new( self.board_size )
        self.moves_legal.count = copy_state.moves_legal.count
        # copy all legal moves from copy_state
        memcpy( self.moves_legal.locations, copy_state.moves_legal.locations, copy_state.moves_legal.count * sizeof( short ) )

        # create groups_list as groups_list
        # this list will contain all alive groups
        # we do not need to set the theoretical max amount of groups as the list 
        # will be incremented in group_list_add
        self.groups_list = groups_list_new( self.board_size )

        # create Group pointer array ( Group **)
        # this array represent the board, every group contains colour, stone-locations
        # and liberty locations
        # border location is included, therefore the array size is board_size +1
        self.board_groups    = <Group **>malloc( ( self.board_size + 1 ) * sizeof( Group* ) )
        if not self.board_groups:
            raise MemoryError()

        # copy all group pointers from copy_state
        # all Groups will be duplicated and overwritten but all group_empty pointers stay the same
        memcpy( self.board_groups, copy_state.board_groups, ( self.board_size + 1 ) * sizeof( Group* ) )

        # loop over all groups in copy_state.groups_list
        # duplicate them and set all Group pointers of this groups stone-locations
        # to the new group
        for i in range( copy_state.groups_list.count_groups ):

            # get group
            group = copy_state.groups_list.board_groups[ i ]             

            # duplicate group
            group_pointer = group_duplicate( group, self.board_size )
            # add new group to groups_list
            groups_list_add( group_pointer, self.groups_list )

            # loop over all group locations
            for location in range( self.board_size ):

                # if group has a stone on this location, set board_groups group pointer 
                if group.locations[ location ] == _STONE:

                    self.board_groups[ location ] = group_pointer


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def __init__( self, char size = 19, GameState copyState = None ):
        """
           create new instance of GameState
        """

        if copyState is not None:

            # create copy of given state
            self.initialize_duplicate( copyState )
        else:

            # check if neighbor arrays exist or size has changed
            if not neighbor or size != neighbor_size:


                # calculate board size
                self.board_size = size * size

                # set globals so they can be changed
                global neighbor
                global neighbor3x3
                global neighbor12d
                global neighbor_size
                global group_empty
                global group_border

                # TODO if size has changed, free all globals

                # set size
                neighbor_size = size

                # set neighbor arrays
                neighbor    = get_neighbors(     size )
                neighbor3x3 = get_3x3_neighbors( size )
                neighbor12d = get_12d_neighbors( size )

                # initialize EMPTY and BORDER group
                group_empty  = group_new( _EMPTY,  self.board_size )
                group_border = group_new( _BORDER, self.board_size )

            # create new root state
            self.initialize_new( size )


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def __dealloc__(self):
        """
           this function is called when this object is destroyed

           Prevent memory leaks by freeing all arrays created with malloc

           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
           do not fee neighbor, neighbor3x3, neighbor12d, 
                      group_empty or group_border

                    RootState will handle those!
           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """

        cdef int i

        # free hash3x3
        if self.hash3x3 is not NULL:

            free( self.hash3x3 )

        # free board_groups
        if self.board_groups is not NULL:

            free( self.board_groups )

        # free history
        locations_list_destroy( self.moves_history )

        # free moves_legal and moves_legal.locations
        if self.moves_legal is not NULL:

            if self.moves_legal.locations is not NULL:

                free( self.moves_legal.locations )

            free( self.moves_legal )

        # free groups_list all groups in groups_list.board_groups and groups_list.board_groups
        if self.groups_list is not NULL:

            # loop over all groups and free them
            for i in range( self.groups_list.count_groups ):

                group_destroy( self.groups_list.board_groups[ i ] )

            # free groups_list.board_groups
            if self.groups_list.board_groups is not NULL:

                free( self.groups_list.board_groups )

            free( self.groups_list )


    ############################################################################
    #   private cdef functions used for game-play                              #
    #                                                                          #
    ############################################################################


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef bint is_legal_move( self, short location, Group **board, short ko ):
        """
           check if playing at location is a legal move to make
        """

        # check if it is empty
        if board[ location ].colour != _EMPTY:
            return 0

        # check ko
        if location == ko:
            return 0

        # check if it has liberty after
        if 0 == self.has_liberty_after( location, board ):
            return 0

        # TODO check super-ko

        return 1


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef bint has_liberty_after( self, short location, Group **board ):
        """
           check if a play at location results in an alive group
           - has liberty
           - conects to group with >= 2 liberty
           - captures enemy group
        """

        cdef int    i
        cdef char   board_value
        cdef short  count_liberty
        cdef short  neighbor_location
        cdef Group* group_temp

        # loop over all four neighbors
        for i in range( 4 ):

            # get neighbor location
            neighbor_location = self.neighbor[ location * 4 + i ]
            board_value       = board[ neighbor_location ].colour

            # if empty location -> liberty -> legal move
            if board_value == _EMPTY:

                return 1

            # get neighbor group
            # ( group_border has zero libery and is wrong colour )
            group_temp    = board[ neighbor_location ]
            count_liberty = group_temp.count_liberty

            # if there is a player_current group
            if board_value == self.player_current:
                
                # if it has at least 2 liberty
                if count_liberty >= 2:

                    # this move removes a liberty
                    # if group has >2 liberty -> legal move
                    return 1

            # if is a player_opponent group and has only one liberty
            elif board_value == self.player_opponent and count_liberty == 1:

                # group killed and thus legal
                return 1
            
        return 0
    

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
    cdef tuple calculate_tuple_location( self, short location ):
        """
           return location on board as a tupple
           no checks on outside board
        """

        # return board location
        return ( location / self.size, location % self.size )


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef void set_moves_legal_list( self, Locations_List *moves_legal ):
        """
           generate moves_legal list
        """

        cdef short i

        # reset moves_legal count
        moves_legal.count = 0

        # TODO? keep empty locations list?
        # loop over all board locations and check if a move is legal
        for i in range( self.board_size ):

            # check if a move is legal
            if self.is_legal_move( i, self.board_groups, self.ko ):

                # add to moves_legal
                moves_legal.locations[ moves_legal.count ] = i
                moves_legal.count += 1


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef void combine_groups( self, Group* group_keep, Group* group_remove, Group **board ):
        """
           combine group_keep and group_remove and replace group_remove on the board 
        """

        cdef int  i
        cdef char value

        # loop over all board locations
        for i in range( self.board_size ):

            value = group_remove.locations[ i ]

            if value == _STONE:

                # group_remove has a stone, add to group_keep
                # and set board location to group_keep
                group_add_stone( group_keep, i )
                board[ i ] = group_keep
            elif value == _LIBERTY:

                # add liberty
                group_add_liberty( group_keep, i )      


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef void remove_group( self, Group* group_remove, Group **board, short* ko ):
        """
           remove group from board -> set all locations to group_empty
        """

        cdef short  location
        cdef short  neighbor_location
        cdef Group* group_temp
        cdef char   board_value
        cdef int    i

        # if groupsize == 1, possible ko
        if group_remove.count_stones == 1:

            ko[ 0 ] = group_location_stone( group_remove, self.board_size )

        # loop over all group stone locations
        for location in range( self.board_size ):

            if group_remove.locations[ location ] == _STONE:

                # set location to empty group
                board[ location ] = self.group_empty

                # update liberty of neighbors
                # loop over all four neighbors
                for i in range( 4 ):

                    # get neighbor location
                    neighbor_location = self.neighbor[ location * 4 + i ]

                    # only current_player groups can be next to a killed group
                    # check if there is a group
                    board_value = board[ neighbor_location ].colour
                    if board_value == self.player_current:

                        # add liberty
                        group_temp = board[ neighbor_location ]
                        group_add_liberty( group_temp, location )


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef void update_hashes( self, Group* group ):
        """
           update all locations affected by removal of group
        """

        cdef short i, a, location, location_array

        # loop over all stones in group
        for i in range( self.board_size ):

            if group.locations[ i ] == _STONE:

                # update hash location
                self.hash3x3[ i ] = self.generate_3x3_hash( i )

                location_array = i * 8

                # loop over diagonal
                # this group is killed -> neighbors are enemy stones
                for a in range( 4, 8 ):

                    # TODO add check for this group location
                    location = self.neighbor3x3[ location_array + a ]
                    if self.board_groups[ location ].colour == _EMPTY:

                        self.hash3x3[ location ] = self.generate_3x3_hash( location )


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef void add_to_group( self, short location, Group **board, short* ko, short* count_captures ):
        """
           check if a stone on location is connected to a group, kills a group 
           or is a new group on the board
        """

        cdef Group* newGroup = NULL
        cdef Group* tempGroup
        cdef Group* changes
        cdef short neighborLocation, location_array
        cdef char  boardValue
        cdef char  group_removed = 0
        cdef int   i

        # loop over all four neighbors
        for i in range( 4 ):

            # get neighbor location and value
            neighborLocation = self.neighbor[ location * 4 + i ]
            boardValue       = board[ neighborLocation ].colour

            # check if neighbor is friendly stone
            if boardValue == self.player_current:

                # check if this is the first friendly neighbor we found
                if newGroup is NULL:

                    # first friendly neighbor
                    newGroup = board[ neighborLocation ]
                else:

                    # another friendly group, if they are different combine them
                    tempGroup = board[ neighborLocation ]
                    if tempGroup != newGroup:

                        self.combine_groups( newGroup, tempGroup, board )

                        # remove temp_group from groupList and destroy it
                        groups_list_remove( tempGroup, self.groups_list )
                        group_destroy( tempGroup )

            elif boardValue == self.player_opponent:

                # remove liberty from enemy group
                tempGroup = board[ neighborLocation ]
                group_remove_liberty( tempGroup, location )

                # check liberty count and remove if 0
                if tempGroup.count_liberty == 0:

                    # increment capture count
                    count_captures[ 0 ] += tempGroup.count_stones

                    # remove group and update hashes
                    self.remove_group( tempGroup, board, ko )
                    self.update_hashes( tempGroup )
                    # TODO hashes of locations next to a group where liberty change also have to be updated

                    # remove tempGroup from groupList and destroy
                    groups_list_remove( tempGroup, self.groups_list )
                    group_destroy( tempGroup )

                    # increment group_removed count
                    group_removed += 1

        # check if no connected group is found
        if newGroup is NULL:    

            # create new group and add to groups_list
            newGroup = group_new( self.player_current, self.board_size )
            groups_list_add( newGroup, self.groups_list )
        else:

            # remove liberty from group
            group_remove_liberty( newGroup, location )

        # add stone to group
        group_add_stone( newGroup, location )
        # set board location to group
        board[ location ] = newGroup

        # calculate location in neighbor array
        location_array = location * 8

        # loop over all four neighbors
        for i in range( 4 ):

            # get neighbor location
            neighborLocation = self.neighbor3x3[ location_array + i ]

            # if neighbor location is empty add liberty and update hash
            if board[ neighborLocation ].colour == _EMPTY:

                group_add_liberty( newGroup, neighborLocation )
                self.hash3x3[ neighborLocation ] = self.generate_3x3_hash( neighborLocation )

        # loop over all four diagonals
        for i in range( 4, 8 ):

            # get neighbor location
            neighborLocation = self.neighbor3x3[ location_array + i ]

            # if diagonal is empty update hash
            if board[ neighborLocation ].colour == _EMPTY:

                self.hash3x3[ neighborLocation ] = self.generate_3x3_hash( neighborLocation )
 
        # check if there is really a ko
        # if two groups died there is no ko
        # if newGroup has more than 1 stone there is no ko
        if group_removed >= 2 or newGroup.count_stones > 1:
             ko[ 0 ] = _PASS


    ############################################################################
    #   private cdef functions used for feature generation                     #
    #                                                                          #
    ############################################################################


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef long generate_12d_hash( self, short centre ):
        """
           generate 12d hash around centre location
        """

        cdef int    i
        cdef long   hash = _HASHVALUE
        cdef Group* group

        # calculate location in neighbor12d array
        centre *= 12

        # hash colour and liberty of all locations
        for i in range( 12 ):

            # get group
            group = self.board_groups[ self.neighbor12d[ centre + i ] ]

            # hash colour
            hash += group.colour
            hash *= _HASHVALUE

            # hash liberty
            hash += min( group.count_liberty, 3 )
            hash *= _HASHVALUE

        return hash


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef long generate_3x3_hash( self, short centre ):
        """
           generate 3x3 hash around centre location
        """

        cdef int    i
        cdef long   hash = _HASHVALUE
        cdef Group* group

        # calculate location in neighbor3x3 array
        centre *= 8

        # hash colour and liberty of all locations
        for i in range( 8 ):

            # get group
            group = self.board_groups[ self.neighbor3x3[ centre + i ] ]

            # hash colour
            hash += group.colour
            hash *= _HASHVALUE

            # hash liberty
            hash += min( group.count_liberty, 3 )
            hash *= _HASHVALUE

        return hash

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef void get_group_after( self, char* groups_after, char* locations, char* captures, short location ):
        """
           groups_after is a board_size * 3 array representing STONES, LIBERTY, CAPTURE for every location

           calculate group after a play on location and set
           groups_after[ location * 3 +   ] to stone   count
           groups_after[ location * 3 + 1 ] to liberty count
           groups_after[ location * 3 + 2 ] to capture count
        """

        cdef short       neighbor_location
        cdef short       temp_location
        cdef char        board_value
        cdef Group*      temp_group
        cdef int         i, a
        cdef int         location_array = location * 3
        cdef short       stones, liberty, capture

        # loop over all four neighbors
        for i in range( 4 ):

            # get neighbor location and value
            neighbor_location = self.neighbor[ location * 4 + i  ]
            temp_group        = self.board_groups[ neighbor_location ]
            board_value       = temp_group.colour

            # check if neighbor is friendly stone
            if board_value == _EMPTY:

                locations[ neighbor_location ] = _LIBERTY
            elif board_value == self.player_current:

                # found friendly group
                for a in range( self.board_size ):

                    if temp_group.locations[ a ] != _FREE: 

                        locations[ a ] = temp_group.locations[ a ]

            elif board_value == self.player_opponent:

                # get enemy group
                # if it has one liberty it wil be killed -> add potential liberty
                if temp_group.count_liberty == 1:

                    for a in range( self.board_size ):

                        if temp_group.locations[ a ] == _STONE: 

                            captures[ a ] = _CAPTURE

        # add stone
        locations[ location ] = _STONE

        for neighbor_location in range( self.board_size ):

            if captures[ neighbor_location ] == _CAPTURE:

                # loop over all four neighbors
                for i in range( 4 ):

                    # get neighbor location and value
                    temp_location = self.neighbor[ neighbor_location * 4 + i ]
                    if temp_location < self.board_size and locations[ temp_location ] == _STONE:

                        locations[ neighbor_location ] = _LIBERTY
        

        # remove location as liberty
        locations[ location ] = _STONE

        stones  = 0 
        liberty = 0
        capture = 0

        # count all values
        for i in range( self.board_size ):

            if locations[ i ] == _STONE:

                stones += 1
            elif locations[ i ] == _LIBERTY:

                liberty += 1
            if captures[ i ] == _CAPTURE:

                capture += 1
        
        # check max
        if stones > 100:
            stones  = 100 

        if liberty > 100:
            liberty  = 100 

        if capture > 100:
            capture  = 100

        # set values 
        groups_after[ location_array     ] = stones 
        groups_after[ location_array + 1 ] = liberty
        groups_after[ location_array + 2 ] = capture


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef bint is_true_eye( self, short location, Locations_List* eyes, char owner ):
        """
           check if location is a real eye
        """

        cdef int   i
        cdef int   eyes_lenght = eyes.count
        cdef char  board_value, max_bad_diagonal
        cdef char  count_bad_diagonal = 0
        cdef char  count_border = 0
        cdef short location_neighbor
        cdef Locations_List* empty_diag
        
        # TODO benchmark what is faster? first dict lookup then neighbor check or other way around

        if eyes_lenght > 70:
            print "pretty big" + str( eyes_lenght )

        # check if it is a known eye
        for i in range( eyes.count ):

            if location == eyes.locations[ i ]:

                return 1

        # loop over neighbor
        for i in range( 4 ):

            location_neighbor = self.neighbor3x3[ location * 8 + i ]
            board_value       = self.board_groups[ location_neighbor ].colour

            if board_value == _BORDER:

                count_border += 1
            elif not board_value == owner:

                # empty location or enemy stone
                return 0

        empty_diag = locations_list_new( 4 )

        # loop over diagonals
        for i in range( 4, 8 ):

            location_neighbor = self.neighbor3x3[ location * 8 + i ]
            board_value       = self.board_groups[ location_neighbor ].colour

            if board_value == _EMPTY:

                #locations_list_add_location( empty_diag, location_neighbor )
                empty_diag.locations[ empty_diag.count ] = location_neighbor
                empty_diag.count += 1
                count_bad_diagonal += 1
            elif board_value == _BORDER:

                count_border += 1
            elif board_value != owner:

                # enemy stone
                count_bad_diagonal += 1

        # assume location is an eye
        locations_list_add_location_increment( eyes, location )
        #eyes.locations[ eyes.count ] = location
        #eyes.count += 1

        max_bad_diagonal = 1 if count_border == 0 else 0

        if count_bad_diagonal <= max_bad_diagonal:

            # one bad diagonal is allowed in the middle
            locations_list_destroy( empty_diag )
            return 1

        for i in range( empty_diag.count ):

            location_neighbor = empty_diag.locations[ i ]

            if self.is_true_eye( location_neighbor, eyes, owner ):

                count_bad_diagonal -= 1

        locations_list_destroy( empty_diag )

        if count_bad_diagonal <= max_bad_diagonal:

            return 1

        # not an eye
        eyes.count = eyes_lenght
        return 0

           
    ############################################################################
    #   private cdef Ladder functions                                          #
    #                                                                          #
    ############################################################################

    """
       Ladder evaluation consumes a lot of time duplicating data, the original
       version (still can be found in go_python.py) made a copy of the whole
       GameState for every move played.

       This version only duplicates self.board_groups ( so the list with pointers to groups )
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


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef Groups_List* add_ladder_move( self, short location, Group **board, short* ko ):
        """
           create a new group for location move and add all connected groups to it

           similar to add_to_group except no groups are changed or killed and a list
           with groups removed is returned so the board can be restored to original
           position
        """

        # create Group_List able to hold up to 4 changed/removed groups
        cdef Groups_List* removed_groups = groups_list_new( 4 )

        # ko is a pointer -> add [ 0 ] to acces the actual value
        ko[ 0 ] = _PASS

        # play move at location and add removed groups to removed_groups list
        self.get_removed_groups( location, removed_groups, board, ko )

        # change player colour
        self.player_current  = self.player_opponent
        self.player_opponent = ( _BLACK if self.player_current == _WHITE else _WHITE )

        return removed_groups


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef void undo_ladder_move( self, short location, Groups_List* removed_groups, short removed_ko, Group **board, short* ko ):
        """
           Use removed_groups list to return board state to be the same as before
           add_ladder_move was used
        """

        cdef short  i, b, location_neighbor
        cdef Group* group 
        cdef Group* group_remove = board[ location ]

        # reset ko to old value
        # ko is a pointer -> add [ 0 ] to acces the actual value
        ko[ 0 ] = removed_ko

        # change player colour
        self.player_current  = self.player_opponent
        self.player_opponent = ( _BLACK if self.player_current == _WHITE else _WHITE )

        # undo move set location to empty group
        board[ location ] = self.group_empty

        # undo group removals
        for i in range( removed_groups.count_groups ):

            # do group unremovals in reversed order!!!
            # this is important in order to get correct liberty counts
            group = removed_groups.board_groups[ removed_groups.count_groups - i - 1 ]

            # check group colour and determine what happened
            # player_current  -> groups have been combined, set board locations to group
            # player_opponent -> groups have been removed, unremove them
            if group.colour == self.player_opponent:

                # opponent group was removed from the board -> unremove it
                self.unremove_group( group, board )
            else:

               # set all board_groups locations to group
               # liberty have not been changed
               for b in range( self.board_size ):

                   if group.locations[ b ] == _STONE:

                       board[ b ] = group

        # add liberty to neighbor groups
        for i in range( 4 ):

            location_neighbor = self.neighbor[ location * 4 + i ]
            if board[ location_neighbor ].colour > _EMPTY:

                group_add_liberty( board[ location_neighbor ], location )

        # destroy group
        group_destroy( group_remove )

        # free removed_groups
        if removed_groups is not NULL:

            if removed_groups.board_groups is not NULL:

                free( removed_groups.board_groups )

            free( removed_groups )


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef void unremove_group( self, Group* group_unremove, Group **board ):
        """
           unremove group from board
           loop over all stones in this group and set board to group_unremove
           remove liberty from neigbor locations
        """

        cdef short  location
        cdef short  neighbor_location
        cdef Group* group_temp
        cdef int    i

        # loop over all group stone locations
        for location in range( self.board_size ):

            # check if this has a stone on location
            if group_unremove.locations[ location ] == _STONE:

                # set location to group_unremove
                board[ location ] = group_unremove

                # update liberty of neighbors
                # loop over all four neighbors
                for i in range( 4 ):

                    # get neighbor location
                    neighbor_location = self.neighbor[ location * 4 + i ]

                    # only current_player groups can be next to a killed group
                    # check if neighbor_location does not belong to this group
                    if group_unremove.locations[ neighbor_location ] != _STONE:

                        # remove liberty
                        group_remove_liberty( board[ neighbor_location ], location )


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef dict get_capture_moves( self, Group* group, char color, Group **board ):
        """
           create a dict with al moves that capture a group surrounding group
        """

        cdef int i, location, location_neighbor, location_array
        cdef Group* group_neighbor
        cdef dict capture = {}

        # find all moves capturing an enemy group
        for location in range( self.board_size ):

            if group.locations[ location ] == _STONE:

                # calculate array location
                location_array = location * 4

                # loop over neighbor
                for i in range( 4 ):

                    # calculate neighbor location
                    location_neighbor = self.neighbor[ location_array + i ]

                    # if location has opponent stone
                    if board[ location_neighbor ].colour == color:

                        # get opponent group
                        group_neighbor = board[ location_neighbor ]

                        # if liberty count == 1
                        if group_neighbor.count_liberty == 1:

                            # add potential capture move
                            location_neighbor = group_location_liberty( group_neighbor, self.board_size )
                            capture[ location_neighbor ] = location_neighbor

        return capture


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef void get_removed_groups( self, short location, Groups_List* removed_groups, Group **board, short* ko ):
        """
           create a new group for location move and add all connected groups to it

           similar to add_to_group except no groups are changed or killed 
           all changes to the board are stored in removed_groups
        """

        # create new group ( it is not added to groups_list as in add_to_group )
        cdef Group* newGroup = group_new( self.player_current, self.board_size )
        cdef Group* tempGroup
        cdef short neighborLocation
        cdef char  boardValue
        cdef char  group_removed = 0
        cdef int   i

        # loop over all four neighbors
        for i in range( 4 ):

            # get neighbor location and value
            neighborLocation = self.neighbor[ location * 4 + i ]
            boardValue       = board[ neighborLocation ].colour

            # check if neighbor is friendly stone
            if boardValue == self.player_current:

                # another friendly group, if they are different combine them
                tempGroup = board[ neighborLocation ]
                if tempGroup != newGroup:

                    self.combine_groups( newGroup, tempGroup, board )
                    # add tempGroup to removed_groups
                    groups_list_add( tempGroup, removed_groups )

            elif boardValue == self.player_opponent:

                # remove liberty from enemy group
                tempGroup = board[ neighborLocation ]
                group_remove_liberty( tempGroup, location )

                # remove group
                if tempGroup.count_liberty == 0:

                    self.remove_group( tempGroup, board, ko )
                    # add tempGroup to removed_groups
                    groups_list_add( tempGroup, removed_groups )

                    # increment group_removed count
                    group_removed += 1

        # remove liberty
        group_remove_liberty( newGroup, location )

        # add stone
        group_add_stone( newGroup, location )

        # set location to newGroup
        board[ location ] = newGroup

        # loop over all four neighbors
        for i in range( 4 ):

            # get neighbor location
            neighborLocation = self.neighbor[ location * 4 + i ]

            # check is neighbor is empty, add liberty if so
            if board[ neighborLocation ].colour == _EMPTY:

                group_add_liberty( newGroup, neighborLocation )
 
        # check if there is really a ko
        # if two groups died there is no ko
        # if newGroup has more than 1 stone there is no ko
        if group_removed >= 2 or newGroup.count_stones > 1:
             ko[ 0 ] = _PASS


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef bint is_ladder_escape_move( self, Group **board, short* ko, short location_group, dict capture, short location, int maxDepth, char colour_group, char colour_chase ):
        """
           play a ladder move on location, check if group has escaped,
           if the group has 2 liberty it is undetermined ->
           try to capture it by playing at both liberty
        """

        cdef int    i
        cdef short  ko_value
        cdef bint   result
        cdef Group* group
        cdef Group* group_capture
        cdef dict   capture_copy
        cdef Groups_List* removed_groups
        cdef short  location_neighbor, location_stone

        # check if max exploration depth has been reached
        if maxDepth <= 0:
            return 0

        # check if move is legal
        if not self.is_legal_move( location, board, ko[ 0 ] ):

            return 0

        # do ladder move and save ko location
        ko_value       = ko[ 0 ]
        removed_groups = self.add_ladder_move( location, board, ko )

        # check group liberty
        group = board[ location_group ]
        i = group.count_liberty
        if i < 2:

            # no escape
            result = 0
        elif i > 2:

            # escape
            result = 1
        else:

            # 2 liberty, fate undetermined

            # TODO now we have to walk over all locations, somehow let do_ladder_move
            # do this -> saves computation time

            # find all moves capturing an enemy group
            for location_stone in range( self.board_size ):

                if group.locations[ location_stone ] == _STONE:

                    # loop over neighbor
                    for i in range( 4 ):

                        # calculate neighbor location
                        location_neighbor = self.neighbor[ location_stone * 4 + i ]

                        # if location has opponent stone
                        if board[ location_neighbor ].colour == colour_chase:

                            # get opponent group
                            group_capture = board[ location_neighbor ]

                            # if liberty count == 1
                            if group_capture.count_liberty == 1:

                                # add potential capture move
                                location_neighbor = group_location_liberty( group_capture, self.board_size )
                                capture[ location_neighbor ] = location_neighbor

            # try to catch group by playing at one of the two liberty locations
            for location_neighbor in range( self.board_size ):

                if group.locations[ location_neighbor ] == _LIBERTY:

                    if self.is_ladder_capture_move( board, ko,  location_group, capture.copy(), location_neighbor, maxDepth - 1, colour_group, colour_chase ):

                        # undo move
                        self.undo_ladder_move( location, removed_groups, ko_value, board, ko )
                        return 0

            # escaped
            result = 1

        # undo move
        self.undo_ladder_move( location, removed_groups, ko_value, board, ko )

        # return result
        return result


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef bint is_ladder_capture_move( self, Group **board, short* ko, short location_group, dict capture, short location, int maxDepth, char colour_group, char colour_chase ):
        """
           play a ladder move on location, try capture and escape moves
           and see if the group is able to escape ladder
        """

        cdef short  i
        cdef short  ko_value
        cdef Group* group
        cdef dict   capture_copy
        cdef short  location_next
        cdef Groups_List* removed_groups

        # if we haven't found a capture by a certain number of moves, assume it's worked.
        if maxDepth <= 0:

            return 1

        if not self.is_legal_move( location, board, ko[ 0 ] ):

            return 0

        ko_value       = ko[ 0 ]
        removed_groups = self.add_ladder_move( location, board, ko )

        # check if the group at location can be captured
        group = board[ location ]
        if group.count_liberty == 1:

            i = group_location_liberty( group, self.board_size )
            capture[ i ] = i

        # try a capture move
        for location_next in capture:

            capture_copy = capture.copy()
            capture_copy.pop( location_next )
            if self.is_ladder_escape_move( board, ko, location_group, capture.copy(), location_next, maxDepth - 1, colour_group, colour_chase ):

                # undo move
                self.undo_ladder_move( location, removed_groups, ko_value, board, ko )
                return 0

        group = board[ location_group ]

        # try an escape move
        for location_next in range( self.board_size ):

            if group.locations[ location_next ]  == _LIBERTY:

                capture_copy = capture.copy()
                if location_next in capture_copy:
                    capture_copy.pop( location_next )
                if self.is_ladder_escape_move( board, ko, location_group, capture.copy(), location_next, maxDepth - 1, colour_group, colour_chase ):

                    # undo move
                    self.undo_ladder_move( location, removed_groups, ko_value, board, ko )
                    return 0

        # no ladder escape found -> group is captured
        # undo move
        self.undo_ladder_move( location, removed_groups, ko_value, board, ko )
        return 1


    ############################################################################
    #   public cdef functions used by preprocessing                            #
    #                                                                          #
    ############################################################################


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef char* get_groups_after( self ):
        """
           return a short array of size board_size * 3 representing 
           STONES, LIBERTY, CAPTURE for every board location

           max count values are 100

           loop over all legal moves and determine stone count, liberty count and
           capture count of a play on that location
        """

        cdef short  i, location

        # initialize groups_after array
        cdef char *groups_after = <char *>malloc( self.board_size * 3 * sizeof( char ) )
        if not groups_after:
            raise MemoryError()

        #memset( groups_after, 0, self.board_size * 3 * sizeof( char ) )

        # create locations dictionary
        cdef char  *locations    = <char  *>malloc( self.board_size * sizeof( char ) )
        if not locations:
            raise MemoryError()

        # create captures dictionary
        cdef char  *captures     = <char  *>malloc( self.board_size * sizeof( char ) )
        if not captures:
            raise MemoryError()

        # create groups for all legal moves
        for location in range( self.moves_legal.count ):
            
            # initialize both dictionaries to _FREE
            memset( locations, _FREE, self.board_size * sizeof( char ) )
            memset( captures,  _FREE, self.board_size * sizeof( char ) )

            self.get_group_after( groups_after, locations, captures, self.moves_legal.locations[ location ] )

        free( locations )
        free( captures )

        return groups_after


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef list get_neighbor_locations( self ):
        """
           generate list with 3x3 neighbor locations
           0,1,2,3 are direct neighbor
           4,5,6,7 are diagonal neighbor
           where -1 if it is a border location or non empty location
        """

        return []


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef long get_hash_12d( self, short centre ):
        """
           return hash for 12d star pattern around location
        """

        # get 12d hash value and add current player colour

        return self.generate_12d_hash( centre ) + self.player_current


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef long get_hash_3x3( self, short location ):
        """
           return 3x3 pattern hash + current player
        """

        # 3x3 hash patterns are updated every move
        # get 3x3 hash value and add current player colour

        return self.hash3x3[ location ] + self.player_current


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef char* get_ladder_escapes( self, int maxDepth ):
        """
           return char array with size board_size
           every location represents a location on the board where:
           _FREE  = no ladder escape
           _STONE = ladder escape           
        """

        cdef short   i, location_group, location_move
        cdef Group*  group
        cdef dict    move_capture
        cdef dict    move_capture_copy
        cdef Group **board   = NULL
        cdef short   ko      = self.ko

        # create char array representing the board
        cdef char*   escapes = <char *>malloc( self.board_size )
        if not escapes:
            raise MemoryError()
        # set all locations to _FREE
        memset( escapes, _FREE, self.board_size )

        # loop over all groups on board
        for i in range( self.groups_list.count_groups ):

            group = self.groups_list.board_groups[ i ]

            # get liberty count
            if group.count_liberty == 1:

                # check if group has one liberty and is owned by current
                if group.colour == self.player_current:

                    # the first time a possible ladder location is found, board
                    # is duplicated, technically this is not neccisary but it is
                    # safer when we start using a multi threaded mcts
                    if board is NULL:

                        # create new Group pointer array as board
                        board = <Group **>malloc( ( self.board_size + 1 ) * sizeof( Group* ) )
                        if not self.board_groups:
                            raise MemoryError()

                        # as the ladder search does not change any excisting groups we can safely
                        # duplicate the whole pointer array without duplicating all groups
                        memcpy( board, self.board_groups, ( self.board_size + 1 ) * sizeof( Group* ) )

                    # get a dictionary with all possible capture groups -> surrounding groups 
                    # the ladder group can kill in order to escape ladder
                    move_capture = self.get_capture_moves( group, self.player_opponent, board )
                    location_group = group_location_stone( group, self.board_size )

                    # check if any of the moves is an escape move
                    for location_move in range( self.board_size ):

                        if group.locations[ location_move ] == _LIBERTY and escapes[ location_move ] == _FREE:

                            # check if group can escape ladder by playing move
                            if self.is_ladder_escape_move( board, &ko, location_group, move_capture.copy(), location_move, maxDepth, self.player_current, self.player_opponent ):

                                escapes[ location_move ] = _STONE

                    # check if any of the capture moves is an escape move
                    for location_move in move_capture:

                        if escapes[ location_move ] == _FREE:

                            move_capture_copy = move_capture.copy()
                            move_capture_copy.pop( location_move )

                            # check if group can escape ladder by playing capture move
                            if self.is_ladder_escape_move( board, &ko, location_group, move_capture_copy, location_move, maxDepth, self.player_current, self.player_opponent ):

                                escapes[ location_move ] = _STONE

        # free temporary board
        if board is not NULL:

            free( board )

        return escapes


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef char* get_ladder_captures( self, int maxDepth ):
        """
           return char array with size board_size
           every location represents a location on the board where:
           _FREE  = no ladder capture
           _STONE = ladder capture
        """

        cdef short   i, location_group, location_move
        cdef Group*  group
        cdef dict    move_capture
        cdef Group **board    = NULL
        cdef short   ko       = self.ko

        # create char array representing the board
        cdef char*   captures = <char *>malloc( self.board_size )
        if not captures:
            raise MemoryError()
        # set all locations to _FREE
        memset( captures, _FREE, self.board_size )

        # loop over all groups on board
        for i in range( self.groups_list.count_groups ):

            group = self.groups_list.board_groups[ i ]

            # get liberty count
            if group.count_liberty == 2:

                # check if group is owned by opponent
                if group.colour == self.player_opponent:

                    # the first time a possible ladder location is found, board
                    # is duplicated, technically this is not neccisary but it is
                    # safer when we start using a multi threaded mcts
                    if board is NULL:

                        # create new Group pointer array as board
                        board = <Group **>malloc( ( self.board_size + 1 ) * sizeof( Group* ) )
                        if not self.board_groups:
                            raise MemoryError()

                        # as the ladder search does not change any excisting groups we can safely
                        # duplicate the whole pointer array without duplicating all groups
                        memcpy( board, self.board_groups, ( self.board_size + 1 ) * sizeof( Group* ) )

                    # get a dictionary with all possible capture groups -> surrounding groups 
                    # the ladder group can kill in order to escape ladder
                    move_capture = self.get_capture_moves( group, self.player_current, board )
                    location_group = group_location_stone( group, self.board_size )

                    # loop over all liberty
                    for location_move in range( self.board_size ):

                        if group.locations[ location_move ] == _LIBERTY and captures[ location_move ] == _FREE:

                            # check if move is ladder capture
                            if self.is_ladder_capture_move( board, &ko, location_group, move_capture.copy(), location_move, maxDepth, self.player_opponent, self.player_current ):

                                captures[ location_move ] = _STONE

        # free temporary board
        if board is not NULL:

            free( board )

        return captures


    ############################################################################
    #   public cdef functions used for game play                               #
    #                                                                          #
    ############################################################################


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef void add_move( self, short location ):
        """
           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                      Move should be legal!
           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

           play move on location, move should be legal!

           update player_current, history and moves_legal
        """

        # reset ko
        self.ko = _PASS

        # detemine where captures should be added, black captures -> white stones
        #                                          white captures -> black stones
        # ( probably better to think of it as black stones captured, and white stones captured )
        cdef short* captures = ( &self.capture_white if ( self.player_current == _BLACK ) else &self.capture_black )

        # add move to board
        self.add_to_group( location, self.board_groups, &self.ko, captures )

        # switch player colour
        self.player_current = self.player_opponent
        self.player_opponent = ( _BLACK if self.player_current == _WHITE else _WHITE )

        # add move to history
        locations_list_add_location_increment( self.moves_history, location )

        # set moves_legal
        self.set_moves_legal_list( self.moves_legal )

        # TODO
        # update zobrist


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef GameState new_state_add_move( self, short location ):
        """
           copy this gamestate and play move at location
        """

        # create new gamestate, copy all data of self
        state = GameState( copyState = self )

        # do move
        state.add_move( location )

        return state


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef char get_winner_colour( self, int komi ):
        """
           Calculate score of board state and return player ID (1, -1, or 0 for tie)
           corresponding to winner. Uses 'Area scoring'.

           http://senseis.xmp.net/?Passing#1
        """

        cdef short location
        cdef char  board_value
        cdef int   score_white = komi
        cdef int   score_black = 0

        # lists to keep track of black and white eyes
        cdef Locations_List* eyes_white = locations_list_new( self.board_size )
        cdef Locations_List* eyes_black = locations_list_new( self.board_size )

        # loop over whole board
        for location in range( self.board_size ):

            # get location colour
            board_value = self.board_groups[ location ].colour

            if board_value == _WHITE:

                # white stone
                score_white += 1
            elif board_value == _BLACK:

                # black stone
                score_black += 1
            else:

                # empty location, check if it is an eye for black/white
                if self.is_true_eye( location, eyes_black, _BLACK ):

                    score_black += 1
                elif self.is_true_eye( location, eyes_white, _WHITE ):

                    score_white += 1

        # free eyes_black and eyes_white
        locations_list_destroy( eyes_black )
        locations_list_destroy( eyes_white )
      
        # substract passes
        # http://senseis.xmp.net/?Passing#1
        score_black -= self.passes_black
        score_white -= self.passes_white

        # check if black has won, tie -> white wins
        if score_black > score_white:

            # black wins
            return _BLACK
                
        # white wins
        return _WHITE


    ############################################################################
    #   public def functions used for game play (Python)                       #
    #                                                                          #
    ############################################################################


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def do_move( self, action, color=None ):
        """
           Play stone at action=(x,y).
           If it is a legal move, current_player switches to the opposite color
           If not, an IllegalMove exception is raised
        """

        if action is _PASS:

            locations_list_add_location_increment( self.moves_history, _PASS )

            if self.player_opponent == _BLACK:

                self.passes_black += 1
            else:

                self.passes_white += 1

            # change player colour
            self.player_current = self.player_opponent
            self.player_opponent = ( _BLACK if self.player_current == _WHITE else _WHITE )

            # legal moves have to be recalculated
            self.set_moves_legal_list( self.moves_legal )
            return

        if color is not None:

            self.player_current = color
            self.player_opponent = ( _BLACK if self.player_current == _WHITE else _WHITE )

            # legal moves have to be recalculated
            self.set_moves_legal_list( self.moves_legal )

        cdef int   x, y, i
        cdef short location
        ( x, y ) = action
        location = self.calculate_board_location( y, x )

        # check if move is legal
        if not self.is_legal_move( location, self.board_groups, self.ko ):

            print( self.player_current )
            print( location )
            print( self.get_legal_moves(include_eyes=True) )
            print( "" )
            print( self.get_legal_moves(include_eyes=False) )
            self.printer()
            raise IllegalMove( str( action ) )

        # add move
        self.add_move( location )

        return True


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def get_legal_moves( self, include_eyes = True ):
        """
           return a list with all legal moves ( in/excluding eyes )
        """

        cdef int  i
        cdef list moves = []
        cdef Locations_List* moves_list

        if include_eyes:

            moves_list = self.moves_legal
        else:

            moves_list = self.get_sensible_moves()


        for i in range( moves_list.count ):

            moves.append( self.calculate_tuple_location( moves_list.locations[ i ] ) )

        if not include_eyes:

            # free sensible_moves
            locations_list_destroy( moves_list )

        return moves


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def get_winner( self, char komi = 6 ):
        """
           Calculate score of board state and return player ID ( 1, -1, or 0 for tie )
           corresponding to winner. Uses 'Area scoring'.
        """

        return self.get_winner_colour( komi )


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def place_handicap_stone( self, action, color=_BLACK ):
        """
           add handicap stones given by a list of tuples in list handicap
        """

        cdef short fake_capture

        self.player_current  = color
        self.player_opponent = ( _BLACK if self.player_current == _WHITE else _WHITE )

        cdef char  x, y
        cdef short location
        ( x, y ) = action
        location = self.calculate_board_location( y, x )

        # add move
        self.add_to_group( location, self.board_groups, &self.ko, &fake_capture )

        # set legal moves
        self.set_moves_legal_list(  self.moves_legal )


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def place_handicaps( self, list handicap ):
        """
           TODO save handicap stones list?? -> seems not usefull as we also have to copy them
           add handicap stones given by a list of tuples in list handicap
        """

        cdef char  x, y
        cdef short location
        cdef short fake_capture

        if self.moves_history.count > 0:
            raise IllegalMove("Cannot place handicap on a started game")

        for action in handicap:

            ( x, y ) = action
            location = self.calculate_board_location( y, x )
            self.add_to_group( location, self.board_groups, &self.ko, &fake_capture )

        # active player colour reverses
        self.player_current  = _WHITE
        self.player_opponent = _BLACK
        
        # set legal moves
        self.set_moves_legal_list( self.moves_legal )


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def is_end_of_game( self ):
        """
           
        """
        if self.moves_history.count > 1:

            if self.moves_history.locations[ self.moves_history.count - 1 ] == _PASS and self.moves_history.locations[ self.moves_history.count - 2 ] == _PASS and self.player_current == _WHITE:

                return True

        return False


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def is_legal( self, action ):
        """
           determine if the given action (x,y tuple) is a legal move
           note: we only check ko, not superko at this point (TODO!)
        """

        cdef int   i
        cdef char  x, y
        cdef short location
        ( x, y ) = action

        # check outside board
        if x < 0 or y < 0 or x >= self.size or y >= self.size:
            return False

        # calculate location
        location = self.calculate_board_location( y, x )

        if self.is_legal_move( location, self.board_groups, self.ko ):

            return True

        return False


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def copy( self ):
        """
           get a copy of this Game state
        """

        return GameState( copyState = self )


    ############################################################################
    #   public def functions used for unittests                                #
    #                                                                          #
    ############################################################################

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def get_current_player(self):
        """
           Returns the color of the player who will make the next move.
        """

        return self.player_current


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def set_current_player( self, colour ):
        """
           change current player colour
        """

        self.player_current  = colour
        self.player_opponent = ( _BLACK if self.player_current == _WHITE else _WHITE )


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def get_history( self ):
        """
           return history as a list of tuples
        """

        cdef int   i
        cdef short location
        cdef list  history  = []

        for i in range( self.moves_history.count ):

            location = self.moves_history.locations[ i ]

            if location != _PASS:
                
                history.append( self.calculate_tuple_location( location ) )
            else:

                history.append( _PASS )


        return history


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def get_captures_black( self ):
        """
           return amount of black stones captures
        """

        return self.capture_black


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def get_captures_white( self ):
        """
           return amount of white stones captured
        """

        return self.capture_white


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def get_ko_location( self ):
        """
           return ko location
        """

        if self.ko == _PASS:
            return None

        return self.ko


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def is_board_equal( self, GameState state ):
        """
           verify that self and state board layout are the same
        """

        for x in range(self.board_size):

            if self.board_groups[ x ].colour != state.board_groups[ x ].colour:

                return False

        return True


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def is_liberty_equal( self, GameState state ):
        """
           verify that self and state liberty counts are the same           
        """

        for x in range(self.board_size):

            if self.board_groups[ x ].count_liberty != state.board_groups[ x ].count_liberty:

                return False

        return True


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def is_ladder_escape( self, action ):
        """
           check if playing action is a ladder escape
        """
        value = False

        cdef char  x, y
        cdef short location

        ( x, y ) = action
        location = self.calculate_board_location( y, x )

        cdef char* escapes = self.get_ladder_escapes( 80 )

        if escapes[ location ] != _FREE:

            value = True

        # free escapes
        free( escapes )

        return value


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def is_ladder_capture( self, action ):
        """
           check if playing action is a ladder capture
        """
        value = False

        cdef char  x, y
        cdef short location

        ( x, y ) = action
        location = self.calculate_board_location( y, x )

        cdef char* captures = self.get_ladder_captures( 80 )

        if captures[ location ] != _FREE:

            value = True

        # free captures
        free( captures )

        return value


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def is_eye( self, action, color ):
        """
           check if location action is a eye for player color
        """
        value = False

        cdef char  x, y
        cdef short location

        ( x, y ) = action
        location = self.calculate_board_location( y, x )

        # checking all games in the KGS database found a max of 15eyes in one state
        # 25 seems a safe bet
        cdef Locations_List* eyes           = locations_list_new( 80 )

        if self.is_true_eye( location, eyes, color ):

            value = True

        locations_list_destroy( eyes )

        return value


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def get_liberty( self ):
        """
           get numpy array with all liberty counts
        """

        liberty = np.zeros((self.size, self.size), dtype=np.int)

        for x in range(self.size):

            for y in range(self.size):

                location = self.calculate_board_location( y, x )

                liberty[x, y] = self.board_groups[location].count_liberty

        return liberty


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def get_board( self ):
        """
           get numpy array with board locations
        """

        board = np.zeros((self.size, self.size), dtype=np.int)

        for x in range(self.size):

            for y in range(self.size):

                location = self.calculate_board_location( y, x )

                board[x, y] = self.board_groups[location].colour

        return board


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def get_size( self ):
        """
           return size
        """

        return self.size


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def get_handicap( self ):
        """
           TODO
           return list with handicap stones placed
        """

        return []


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef Locations_List* get_sensible_moves( self ):
        """
           only used for def get_legal_moves
        """

        # TODO validate usage of struct is actually faster

        # create list with at least #moves_legal.count locations
        # there can never be more sensible moves
        cdef Locations_List* sensible_moves = locations_list_new( self.moves_legal.count )

        # checking all games in the KGS database found a max of 17eyes in one state
        # 25 seems a safe bet
        cdef Locations_List* eyes           = locations_list_new( 80 )
        cdef int   i
        cdef short location

        for i in range( self.moves_legal.count ):

            location = self.moves_legal.locations[ i ]

            if not self.is_true_eye( location, eyes, self.player_current ):

                # TODO  find out why locations_list_add_location is 2x slower
                #locations_list_add_location( sensible_moves, location )

                sensible_moves.locations[ sensible_moves.count ] = location
                sensible_moves.count += 1

        locations_list_destroy( eyes )

        return sensible_moves


    ############################################################################
    #   tests                                                                  #
    #                                                                          #
    ############################################################################

    def validate_equal( self, GameState state ):

        cdef int i
        value = True

        for i in range( self.board_size ):

            if self.ko != state.ko:

                print( "ko " + str( self.ko )) 
                return False

            if self.board_groups[ i ].colour != state.board_groups[ i ].colour:

                print( "board " + str( i )) 
                value = False

            if self.board_groups[ i ].count_stones != state.board_groups[ i ].count_stones:

                print( "stones " + str( i )) 
                print( str( self.board_groups[ i ].count_stones ) + " " + str( state.board_groups[ i ].count_stones )) 
                value = False

            if self.board_groups[ i ].count_liberty != state.board_groups[ i ].count_liberty:

                print( "liberty " + str( i ) + " " + str( state.board_groups[ i ].colour ) + " " + str( state.player_current ) ) 
                print( str( self.board_groups[ i ].count_liberty ) + " " + str( state.board_groups[ i ].count_liberty ) ) 
                value = False

        return value

    def printer( self ):
        print( "" )
        for i in range( self.size ):
            A = str( i ) + " "
            for j in range( self.size ):

                B = 0
                if self.board_groups[ j + i * self.size ].colour == _BLACK:
                    B = 'B'
                elif self.board_groups[ j + i * self.size ].colour == _WHITE:
                    B = 'W'
                A += str( B ) + " "
            print( A )


    # do move, throw exception when outside the board
    # action has to be a ( x, y ) tuple
    # this function should be used from Python environment, 
    # use add_move from C environment for speed
    def do_ladder_move( self, action ):
        """
           
        """

        # do move, return true if legal, return false if not

        cdef int   x, y
        cdef short location
        ( x, y ) = action
        location = self.calculate_board_location( y, x )
        self.add_ladder_move( location, self.board_groups, &self.ko )

        self.set_moves_legal_list( self.moves_legal )

        return True


    # do move, throw exception when outside the board
    # action has to be a ( x, y ) tuple
    # this function should be used from Python environment, 
    # use add_move from C environment for speed
    def do_and_undo_ladder_move( self, action ):
        """
           
        """

        # do move, return true if legal, return false if not

        cdef int   x, y
        cdef short location, ko
        ( x, y ) = action
        location = self.calculate_board_location( y, x )

        cdef Groups_List* removed_groups

        ko = self.ko
        removed_groups = self.add_ladder_move( location, self.board_groups, &self.ko )

        self.undo_ladder_move( location, removed_groups, ko, self.board_groups, &self.ko )


        return True

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef test( self ):

        print( "empty" )


    cdef test_cpp_fast( self ):

        print( "empty" )


    def test_cpp( self ):

        print( "cpp" )
        self.test_cpp_fast()
        print( "cpp" )

    def test_game_speed( self, list moves ):

        cdef short location

        for location in moves:
            self.add_move( location )

    def convert_moves( self, list moves ):
        cdef list converted_moves = []
        cdef int   x, y
        cdef short location

        for loc in moves:

            ( x, y ) = loc
            location = self.calculate_board_location( y, x )
            converted_moves.append( location )

        return converted_moves

    def millis( self, start_time ):
       dt = datetime.now() - start_time
       ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
       return ms

    def test_stuff( self ):
        cdef long a, h, i, j, k
        cdef short b, e

        if not neighbor:
            print("NOT")

        for i in range( self.board_size * 4 ):
            if neighbor[ i ] != self.neighbor[ i ]:
                print("NOT EQUAL")
                return

        if not neighbor3x3:
            print("NOT")

        for i in range( self.board_size * 8 ):
            if neighbor3x3[ i ] != self.neighbor3x3[ i ]:
                print("NOT EQUAL")
                return

        if not neighbor12d:
            print("NOT")

        for i in range( self.board_size * 12 ):
            if neighbor12d[ i ] != self.neighbor12d[ i ]:
                print("NOT EQUAL")
                return

        cdef long amount = 1000

        e = 1
        b = 0
        start = datetime.now()

        for h in range( amount ):
            for i in range( amount ):

                for a in range( self.board_size * 12 ):

                    b = neighbor12d[ a ]

                    if neighbor12d[ a ] > 100:
                        e = neighbor12d[ a ]
            if b * e > 0:
                b = 1

        end = datetime.now()
        dt  = end - start
        start = dt.microseconds

        print( "glob " + str( start ) + " " + str( b ) )


        e = 1
        b = 0
        start = datetime.now()

        for h in range( amount ):
            for i in range( amount ):

                for a in range( self.board_size * 12 ):

                    b = self.neighbor12d[ a ]

                    if self.neighbor12d[ a ] > 100:
                        e = self.neighbor12d[ a ]
            if b * e > 0:
                b = 1

        end = datetime.now()
        dt  = end - start
        start = dt.microseconds

        print( "self " + str( start ) + " " + str( b ) )
        

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def set_stuff( self ):

        self.get_sensible_moves()


class IllegalMove(Exception):
    pass
