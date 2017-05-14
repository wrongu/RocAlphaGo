#cython: profile=True
#cython: linetrace=True
import sys
import time
cimport cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset

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

    cdef short capture_black
    cdef short capture_white

    # TODO replace list with struct
    cdef list history
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
    cdef void initialize_duplicate( self, GameState copy_state ):
        """
           Initialize all variables as a copy of copy_state
        """

        cdef int    i
        cdef short  location
        cdef Group* group_pointer
        cdef Group* group

        # !!! do not copy !!! -> these do not need a deep copy as they are static
        self.neighbor        = copy_state.neighbor
        self.neighbor3x3     = copy_state.neighbor3x3
        self.neighbor12d     = copy_state.neighbor12d

        # empty group
        self.group_empty     = copy_state.group_empty

        # pattern dictionary

        # zobrist
      # self.hash_lookup     = copy_state.hash_lookup

        # set values
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

        # copy values
        self.history         = list( copy_state.history )
      # self.previous_hashes = list( copy_state.previous_hashes )

        # create array/list

        self.hash3x3         = <long *>malloc( ( self.board_size     ) * sizeof( long ) )
        if not self.hash3x3:
            raise MemoryError()

        memcpy( self.hash3x3, copy_state.hash3x3, ( self.board_size  ) * sizeof( long ) )

        self.moves_legal            = <Locations_List *>malloc( sizeof( Locations_List ) )
        if not self.moves_legal:
            raise MemoryError()

        self.moves_legal.locations  = <short *>malloc( self.board_size * sizeof( short ) )
        if not self.moves_legal.locations:
            raise MemoryError()

        memcpy( self.moves_legal.locations, copy_state.moves_legal.locations, self.board_size * sizeof( short ) )
        self.moves_legal.count      = copy_state.moves_legal.count

        self.groups_list                   = <Groups_List *>malloc( sizeof( Groups_List ) )
        if not self.groups_list:
            raise MemoryError()

        self.groups_list.board_groups      = <Group **>malloc( self.board_size * sizeof( Group* ) )
        if not self.groups_list.board_groups:
            raise MemoryError()

        self.groups_list.count_groups      = 0

        self.board_groups    = <Group **>malloc( ( self.board_size + 1 ) * sizeof( Group* ) )
        if not self.board_groups:
            raise MemoryError()

        # empty group and border group stay the same, the others will be reset
        memcpy( self.board_groups, copy_state.board_groups, ( self.board_size + 1 ) * sizeof( Group* ) )


        # copy all groups and set groupsList
        for i in range( copy_state.groups_list.count_groups ):

            group = copy_state.groups_list.board_groups[ i ]             

            group_pointer = group_duplicate( group, self.board_size )
            groups_list_add( group_pointer, self.groups_list )

            # loop over all group locations and set group
            for location in range( self.board_size ):

                if group.locations[ location ] == _STONE:

                    self.board_groups[ location ] = group_pointer


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    def __init__( self, char size = 19, GameState copyState = None ):
        """
          d create new instance of GameState
        """

        if copyState is not None:

            # create copy of given state
            self.initialize_duplicate( copyState )


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
        if not self.has_liberty_after( location, board ):
            return 0
        # super-ko

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

            # check empty location -> liberty
            if board_value == _EMPTY:

                return 1

            # get neighbor group
            group_temp    = board[ neighbor_location ]
            count_liberty = group_temp.count_liberty

            # if there is a player_current group
            if board_value == self.player_current:
                
                # if it has at least 2 liberty
                if count_liberty >= 2:

                    return 1

            # if is a player_opponent group and has only one liberty
            elif board_value == self.player_opponent and count_liberty == 1:

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
           return location on board
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

        moves_legal.count = 0
        for i in range( self.board_size ):

            if self.is_legal_move( i, self.board_groups, self.ko ):

                moves_legal.locations[ moves_legal.count ] = i
                moves_legal.count += 1


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef void combine_groups( self, Group* group_keep, Group* group_remove, Group **board ):
        """
           combine two groups and remove one
        """

        cdef int  i
        cdef char value

        # combine stones, liberty and set groups
        for i in range( self.board_size ):

            value = group_remove.locations[ i ]

            if value == _STONE:

                group_add_stone( group_keep, i )
                board[ i ] = group_keep
            elif value == _LIBERTY:

                group_add_liberty( group_keep, i )      


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef void remove_group( self, Group* group_remove, Group **board, short* ko ):
        """
           remove group from board
        """

        cdef short  location
        cdef short  neighbor_location
        cdef Group* group_temp
        cdef char   board_value
        cdef int    i
        # empty group is always in border location
        cdef Group* group_empty = self.group_empty

        # if groupsize == 1, possible ko
        if group_remove.count_stones == 1:

            ko[ 0 ] = group_location_stone( group_remove, self.board_size )

        # loop over all group stone locations
        for location in range( self.board_size ):

            if group_remove.locations[ location ] == _STONE:

                # set location to empty group
                board[ location ] = group_empty

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

        # update all 3x3 hashes in update_hash_locations


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
                for a in range( 4, 8 ):

                    location = self.neighbor3x3[ location_array + a ]
                    if self.board_groups[ location ].colour == _EMPTY:

                        self.hash3x3[ location ] = self.generate_3x3_hash( location )

    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef void add_to_group( self, short location, Group **board, short* ko, short* count_captures ):
        """
           add location to group or create new one
        """

        cdef Group* newGroup = NULL
        cdef Group* tempGroup
        cdef Group* changes
        cdef short neighborLocation, location_array
        cdef char  boardValue
        cdef char  group_removed = 0
        cdef int   i

        # find friendly stones
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
                        # remove temp_group from groupList and destroy
                        groups_list_remove( tempGroup, self.groups_list )
                        group_destroy( tempGroup )

            elif boardValue == self.player_opponent:

                # remove liberty from enemy group
                tempGroup = board[ neighborLocation ]
                group_remove_liberty( tempGroup, location )

                # remove group
                if tempGroup.count_liberty == 0:

                    count_captures[ 0 ] += tempGroup.count_stones
                    self.remove_group( tempGroup, board, ko )
                    self.update_hashes( tempGroup )
                    # remove tempGroup from groupList and destroy
                    groups_list_remove( tempGroup, self.groups_list )
                    group_destroy( tempGroup )
                    group_removed += 1

        # check if a group was found or create one
        if newGroup is NULL:    

            newGroup = group_new( self.player_current, self.board_size )
            groups_list_add( newGroup, self.groups_list )
        else:

            group_remove_liberty( newGroup, location )

        # add stone
        group_add_stone( newGroup, location )
        # set location group
        board[ location ] = newGroup

        location_array = location * 8

        # add new liberty
        # loop over all four neighbors
        for i in range( 4 ):

            # get neighbor location
            neighborLocation = self.neighbor3x3[ location_array + i ]

            if board[ neighborLocation ].colour == _EMPTY:

                group_add_liberty( newGroup, neighborLocation )
                self.hash3x3[ neighborLocation ] = self.generate_3x3_hash( neighborLocation )

        # loop over all four diagonals
        for i in range( 4, 8 ):

            # get neighbor location
            neighborLocation = self.neighbor3x3[ location_array + i ]

            if board[ neighborLocation ].colour == _EMPTY:

                self.hash3x3[ neighborLocation ] = self.generate_3x3_hash( neighborLocation )
 
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
    cdef void get_group_after( self, short* groups_after, char* locations, char* captures, short location ):
        """
           return group as it is after playing at location
        """

        cdef short       neighbor_location
        cdef short       temp_location
        cdef char        board_value
        cdef Group*      temp_group
        cdef int         i, a
        cdef int         location_array = location * 3
        cdef short       stones, liberty, capture

        # find friendly stones
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

        for i in range( self.board_size ):

            if locations[ i ] == _STONE:

                stones += 1
            elif locations[ i ] == _LIBERTY:

                liberty += 1
            if captures[ i ] == _CAPTURE:

                capture += 1
        
        groups_after[ location_array ] = stones 

        location_array += 1
        groups_after[ location_array ] = liberty

        location_array += 1
        groups_after[ location_array ] = capture


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef bint is_true_eye( self, short location, Locations_List* eyes, char owner ):
        """
           
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
        #locations_list_add_location( eyes, location )
        eyes.locations[ eyes.count ] = location
        eyes.count += 1

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


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef Groups_List* add_ladder_move( self, short location, Group **board, short* ko ):
        """
           
        """

        # assume legal move!
        cdef Groups_List* removed_groups

        removed_groups = <Groups_List *>malloc( sizeof( Groups_List ) )
        if not removed_groups:
            raise MemoryError()

        removed_groups.board_groups = <Group **>malloc( 4 * sizeof( Group* ) )
        if not removed_groups.board_groups:
            raise MemoryError()

        removed_groups.count_groups = 0

        ko[ 0 ] = _PASS

        self.get_removed_groups( location, removed_groups, board, ko )

        # change player colour
        self.player_current  = self.player_opponent
        self.player_opponent = ( _BLACK if self.player_current == _WHITE else _WHITE )

        return removed_groups


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef void unremove_group( self, Group* group_unremove, Group **board ):
        """
           unremove group from board
        """

        cdef short  location
        cdef short  neighbor_location
        cdef Group* group_temp
        cdef int    i

        # loop over all group stone locations
        for location in range( self.board_size ):

            if group_unremove.locations[ location ] == _STONE:

                # set location to group_unremove
                board[ location ] = group_unremove

                # update liberty of neighbors
                # loop over all four neighbors
                for i in range( 4 ):

                    # get neighbor location
                    neighbor_location = self.neighbor[ location * 4 + i ]

                    # only current_player groups can be next to a killed group
                    # check if there is a group
                    if board[ neighbor_location ].colour == self.player_current:

                        # remove liberty
                        group_temp = board[ neighbor_location ]
                        group_remove_liberty( group_temp, location )


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef dict get_capture_moves( self, Group* group, char color, Group **board ):
        """
           
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
           add location to group or create new one
        """

        cdef Group* newGroup = group_new( self.player_current, self.board_size )
        cdef Group* tempGroup
        cdef short neighborLocation
        cdef char  boardValue
        cdef char  group_removed = 0
        cdef int   i

        # find friendly stones
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
                    group_removed += 1

        # remove liberty
        group_remove_liberty( newGroup, location )

        # add stone
        group_add_stone( newGroup, location )

        # set location group
        board[ location ] = newGroup

        # add new liberty
        # loop over all four neighbors
        for i in range( 4 ):

            # get neighbor location
            neighborLocation = self.neighbor[ location * 4 + i ]

            if board[ neighborLocation ].colour == _EMPTY:

                group_add_liberty( newGroup, neighborLocation )
 
        # if two groups died there is no ko
        # if newGroup has more than 1 stone there is no ko
        if group_removed >= 2 or newGroup.count_stones > 1:
             ko[ 0 ] = _PASS


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef void undo_ladder_move( self, short location, Groups_List* removed_groups, short removed_ko, Group **board, short* ko ):
        """
           
        """

        cdef short  i, b, location_neighbor
        cdef Group* group 
        cdef Group* group_remove = board[ location ]

        ko[ 0 ] = removed_ko

        # change player colour
        self.player_current  = self.player_opponent
        self.player_opponent = ( _BLACK if self.player_current == _WHITE else _WHITE )

        # set group to empty group
        board[ location ] = self.group_empty

        # undo group removals
        for i in range( removed_groups.count_groups ):

            group = removed_groups.board_groups[ removed_groups.count_groups - i - 1 ]

            if board[ group_location_stone( group, self.board_size ) ].colour == _EMPTY:

                # opponent group was removed
                self.unremove_group( group, board )
            else:

               # set all board_groups locations to group
               for b in range( self.board_size ):

                   if group.locations[ b ] == _STONE:

                       board[ b ] = group

        # add liberty to neighbor groups -> check if needed
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
    cdef bint is_ladder_escape_move( self, Group **board, short* ko, short location_group, dict capture, short location, int maxDepth, char colour_group, char colour_chase ):
        """
           
        """

        cdef int    i
        cdef short  ko_value
        cdef bint   result
        cdef Group* group
        cdef Group* group_capture
        cdef Groups_List* removed_groups
        cdef short  location_neighbor, location_stone

        if maxDepth <= 0:
            return 0

        if not self.is_legal_move( location, board, ko[ 0 ] ):

            return 0

        # do move
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

            # try to capture group by playing at one of the two liberty locations
            for location_neighbor in range( self.board_size ):

                if group.locations[ location_neighbor ] == _LIBERTY:

                    if self.is_ladder_capture_move( board, ko,  location_group, capture.copy(), location_neighbor, maxDepth - 1, colour_group, colour_chase ):

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

        group = board[ location ]
        if group.count_liberty == 1:

            i = group_location_liberty( group, self.board_size )
            capture[ i ] = i

        # try a capture move
        for location_next in capture:

            capture_copy = capture.copy()
            capture_copy.pop( location_next )
            if self.is_ladder_escape_move( board, ko, location_group, capture_copy, location_next, maxDepth - 1, colour_group, colour_chase ):

                self.undo_ladder_move( location, removed_groups, ko_value, board, ko )
                return 0

        group = board[ location_group ]

        # try an escape move
        for location_next in range( self.board_size ):

            if group.locations[ location_next ]  == _LIBERTY:

                if self.is_ladder_escape_move( board, ko, location_group, capture.copy(), location_next, maxDepth - 1, colour_group, colour_chase ):

                    self.undo_ladder_move( location, removed_groups, ko_value, board, ko )
                    return 0

        # no ladder escape found -> group is captured
        self.undo_ladder_move( location, removed_groups, ko_value, board, ko )
        return 1


    ############################################################################
    #   public cdef functions used by preprocessing                            #
    #                                                                          #
    ############################################################################


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef short* get_groups_after( self ):
        """
           
        """

        cdef short  i, location
        cdef short *groups_after = <short *>malloc( self.board_size * 3 * sizeof( short ) )
        if not groups_after:
            raise MemoryError()

        memset( groups_after, 0, self.board_size * 3 * sizeof( short ) )

        cdef char  *locations    = <char  *>malloc( self.board_size * sizeof( char ) )
        if not locations:
            raise MemoryError()

        cdef char  *captures     = <char  *>malloc( self.board_size * sizeof( char ) )
        if not captures:
            raise MemoryError()

        # create groups for all legal moves
        for location in range( self.moves_legal.count ):
            
            memset( locations, _FREE, self.board_size * sizeof( char ) )
            memset( captures,  _FREE, self.board_size * sizeof( char ) )

            self.get_group_after( groups_after, locations, captures, self.moves_legal.locations[ location ] )

        free( locations )
        free( captures )

        return groups_after


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef Locations_List* get_sensible_moves( self ):
        """
           
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

        return self.generate_12d_hash( centre )


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef long get_hash_3x3( self, short location ):
        """
           return 3x3 pattern hash + current player
        """

        # 3x3 hash patterns are updated every move
        # get 3x3 hash value and add current player 

        return self.hash3x3[ location ] + self.player_current


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef char* get_ladder_escapes( self, int maxDepth ):
        """
           
        """

        cdef short   i, location_group, location_move
        cdef Group*  group
        cdef dict    move_capture
        cdef dict    move_capture_copy
        cdef Group **board   = NULL
        cdef short   ko      = self.ko
        cdef char*   escapes = <char *>malloc( self.board_size )
        if not escapes:
            raise MemoryError()

        memset( escapes, _FREE, self.board_size )

        # loop over all groups on board
        for i in range( self.groups_list.count_groups ):

            group = self.groups_list.board_groups[ i ]

            # get liberty count
            if group.count_liberty == 1:

                # check if group has one liberty and is owned by current
                if group.colour == self.player_current:

                    if board is NULL:

                        board = <Group **>malloc( ( self.board_size + 1 ) * sizeof( Group* ) )
                        if not self.board_groups:
                            raise MemoryError()

                        # empty group and border group stay the same, the others will be reset
                        memcpy( board, self.board_groups, ( self.board_size + 1 ) * sizeof( Group* ) )

                    move_capture = self.get_capture_moves( group, self.player_opponent, board )
                    location_group = group_location_stone( group, self.board_size )

                    for location_move in range( self.board_size ):

                        if group.locations[ location_move ] == _LIBERTY and escapes[ location_move ] == _FREE:

                            if self.is_ladder_escape_move( board, &ko, location_group, move_capture.copy(), location_move, maxDepth, self.player_current, self.player_opponent ):

                                escapes[ location_move ] = _STONE

                    for location_move in move_capture:

                        if escapes[ location_move ] == _FREE:

                            move_capture_copy = move_capture.copy()
                            move_capture_copy.pop( location_move )

                            if self.is_ladder_escape_move( board, &ko, location_group, move_capture_copy, location_move, maxDepth, self.player_current, self.player_opponent ):

                                escapes[ location_move ] = _STONE


        if board is not NULL:

            free( board )

        return escapes


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef char* get_ladder_captures( self, int maxDepth ):
        """
           
        """

        cdef short   i, location_group, location_move
        cdef Group*  group
        cdef dict    move_capture
        cdef Group **board    = NULL
        cdef short   ko       = self.ko
        cdef char*   captures = <char *>malloc( self.board_size )
        if not captures:
            raise MemoryError()

        memset( captures, _FREE, self.board_size )

        # loop over all groups on board
        for i in range( self.groups_list.count_groups ):

            group = self.groups_list.board_groups[ i ]

            # get liberty count
            if group.count_liberty == 2:

                # check if group is owned by opponent
                if group.colour == self.player_opponent:

                    if board is NULL:

                        board = <Group **>malloc( ( self.board_size + 1 ) * sizeof( Group* ) )
                        if not self.board_groups:
                            raise MemoryError()

                        # empty group and border group stay the same, the others will be reset
                        memcpy( board, self.board_groups, ( self.board_size + 1 ) * sizeof( Group* ) )


                    move_capture = self.get_capture_moves( group, self.player_current, board )
                    location_group = group_location_stone( group, self.board_size )

                    # loop over liberty
                    for location_move in range( self.board_size ):

                        if group.locations[ location_move ] == _LIBERTY and captures[ location_move ] == _FREE:

                            if self.is_ladder_capture_move( board, &ko, location_group, move_capture.copy(), location_move, maxDepth, self.player_opponent, self.player_current ):

                                captures[ location_move ] = _STONE

        if board is not NULL:

            free( board )

        return captures


    @cython.boundscheck( False )
    @cython.wraparound(  False )
    cdef char get_board_feature( self, short location ):
        """
           return correct board feature value
           - 0 active player stone
           - 1 opponent stone
           - 2 empty location
        """

        cdef char value = self.board_groups[ location ].colour

        if value == _EMPTY:
            return 2

        if value == self.player_current:
            return 0

        return 1


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

        # where captures should be added, black captures white stones
        #                                 white captures black stones
        cdef short* captures = ( &self.capture_white if ( self.player_current == _BLACK ) else &self.capture_black )

        # add move
        self.add_to_group( location, self.board_groups, &self.ko, captures )

        # change player colour
        self.player_current = self.player_opponent
        self.player_opponent = ( _BLACK if self.player_current == _WHITE else _WHITE )

        # add to history
        self.history.append( location )

        # set moves legal
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

        # place move
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
        cdef Locations_List* eyes_white = locations_list_new( self.board_size )
        cdef Locations_List* eyes_black = locations_list_new( self.board_size )

        for location in range( self.size * self.size ):

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

            self.history.append( _PASS )

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

        if len( self.history ) > 0:
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


    def is_end_of_game( self ):
        """
           
        """
        if len( self.history ) > 1:

            if self.history[-1] is _PASS and self.history[-2] is _PASS and self.player_current == _WHITE:

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

        history = []

        for location in self.history:

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

    def test_stuff( self ):

        print( "test stuff" )
        self.test()


class IllegalMove(Exception):
    pass
