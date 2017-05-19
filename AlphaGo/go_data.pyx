cimport cython
from libc.stdlib cimport malloc, free, realloc
from libc.string cimport memcpy, memset, memchr

"""
   Future speedups, right now the usage of C dicts and List is copied from original
   Java implementation. not all usages have been tested for max performance.

   possible speedups could be swapping certain dicts for lists and vice versa.
   more testing should be done where this might apply.

   some notes:
   - using list for Group stone&liberty locations?
   - do we need to consider 25*25 boards?
   - dict for moves_legal instead of list?
   - create mixed short+char arrays to store location+value in one array?
   - implement dict+list struct to get fast lookup and fast looping over all elements
   - store one liberty&stone location in group for fast lookup of group location/liberty
   - implement faster loop over all elements for dict using memchr and offset pointer
"""

############################################################################
#   constants                                                              #
#                                                                          #
############################################################################


# value for PASS move
_PASS    = -1

# observe: stones > EMPTY
#          border < EMPTY
# be aware you should NOT use != EMPTY as this includes border locations
_BORDER  = 1
_EMPTY   = 2
_WHITE   = 3
_BLACK   = 4

# used for group stone, liberty locations, legal move and sensible move
_FREE    = 3
_STONE   = 0
_LIBERTY = 1
_CAPTURE = 2
_LEGAL   = 4
_EYE     = 5

# value used to generate pattern hashes
_HASHVALUE = 33


############################################################################
#   Structs                                                                #
#                                                                          #
############################################################################

""" -> structs, declared in go_data.pxd

# struct to store group information
cdef struct Group:
    char  *locations
    short  count_stones
    short  count_liberty
    char   colour

# struct to store a list of Group
cdef struct Groups_List:
    Group **board_groups
    short   count_groups

# struct to store a list of short ( board locations )
cdef struct Locations_List:
    short  *locations
    short   count
"""

############################################################################
#   group functions                                                        #
#                                                                          #
############################################################################


@cython.boundscheck( False )
@cython.wraparound(  False )
cdef Group* group_new( char colour, short size ):
    """
       create new struct Group
       with locations #size char long initialized to FREE
    """

    cdef int i

    # allocate memory for Group
    cdef Group *group = <Group *>malloc( sizeof( Group ) )
    if not group:
        raise MemoryError()

    # allocate memory for array locations
    group.locations = <char  *>malloc( size )
    if not group.locations:
        raise MemoryError()

    # set counts to 0 and colour to colour
    group.count_stones  = 0
    group.count_liberty = 0
    group.colour        = colour

    # initialize locations with FREE
    memset( group.locations, _FREE, size )

    return group


@cython.boundscheck( False )
@cython.wraparound(  False )
cdef Group* group_duplicate( Group* group, short size ):
    """
       create new struct Group initialized as a duplicate of group
    """

    cdef int i

    # allocate memory for Group
    cdef Group *duplicate = <Group *>malloc( sizeof( Group ) )
    if not duplicate:
        raise MemoryError()

    # allocate memory for array locations
    duplicate.locations = <char  *>malloc( size )
    if not duplicate.locations:
        raise MemoryError()

    # set counts and colour values
    duplicate.count_stones  = group.count_stones
    duplicate.count_liberty = group.count_liberty
    duplicate.colour        = group.colour

    # duplicate locations array in memory
    # memcpy is optimized to do this quickly
    memcpy( duplicate.locations, group.locations, size )

    return duplicate


@cython.boundscheck( False )
@cython.wraparound(  False )
cdef void group_destroy( Group* group ):
    """
       free memory location of group and locations
    """

    # check if group exists
    if group is not NULL:

        # check if locations exists
        if group.locations is not NULL:

            # free locations
            free( group.locations  )

        # free group
        free( group )


@cython.boundscheck( False )
@cython.wraparound(  False )
cdef void group_add_stone( Group* group, short location ):
    """
       update location as STONE
       update liberty count if it was a liberty location

       n.b. stone count is not incremented if a stone was present already
    """

    # check if locations is a liberty
    if group.locations[ location ] == _FREE:

        # locations is FREE, increment stone count
        group.count_stones += 1
    elif group.locations[ location ] == _LIBERTY:

        # locations is LIBERTY, increment stone count and decrement liberty count
        group.count_stones  += 1
        group.count_liberty -= 1

    # set STONE
    group.locations[ location ] = _STONE


@cython.boundscheck( False )
@cython.wraparound(  False )
cdef void group_remove_stone( Group* group, short location ):
    """
       update location as FREE
       update stone count if it was a stone location
    """

    # check if a stone is present
    if group.locations[ location ] == _STONE:

        # stone present, decrement stone count and set location to FREE
        group.count_stones -= 1
        group.locations[ location ] = _FREE


@cython.boundscheck( False )
@cython.wraparound(  False )
cdef short group_location_stone( Group* group, short size ):
    """
       return first location where a STONE is located
    """

    # memchr is a in memory search function, it starts searching at 
    # pointer location #group.locations for a max of size continous bytes untill
    # a location with value _STONE is found -> returns a pointer to this location
    # when this pointer location is substracted with pointer #group.locations
    # the location is calculated where a stone is
    return <short>( <long>memchr( group.locations, _STONE, size ) - <long>group.locations )


@cython.boundscheck( False )
@cython.wraparound(  False )
cdef void group_add_liberty( Group* group, short location ):
    """
       update location as LIBERTY
       update liberty count if it was a FREE location

       n.b. liberty count is not incremented if a stone was present already
    """

    # check if location is FREE
    if group.locations[ location ] == _FREE:

        # increment liberty count, set location to LIBERTY
        group.count_liberty += 1
        group.locations[ location ] = _LIBERTY


@cython.boundscheck( False )
@cython.wraparound(  False )
cdef void group_remove_liberty( Group* group, short location ):
    """
       update location as FREE
       update liberty count if it was a LIBERTY location

       n.b. liberty count is not decremented if location is a FREE location
    """

    # check if location is LIBERTY
    if group.locations[ location ] == _LIBERTY:

        # decrement liberty count, set location to FREE
        group.count_liberty -= 1
        group.locations[ location ] = _FREE


@cython.boundscheck( False )
@cython.wraparound(  False )
cdef short group_location_liberty( Group* group, short size ):
    """
       return location where a LIBERTY is located
    """

    # memchr is a in memory search function, it starts searching at 
    # pointer location #group.locations for a max of size continous bytes untill
    # a location with value _LIBERTY is found -> returns a pointer to this location
    # when this pointer location is substracted with pointer #group.locations
    # the location is calculated where a liberty is
    return <short>( <long>memchr(group.locations, _LIBERTY, size ) - <long>group.locations )


############################################################################
#   Groups_List functions                                                  #
#                                                                          #
############################################################################


@cython.boundscheck( False )
@cython.wraparound(  False )
cdef Groups_List* groups_list_new( short size ):
    """
       create new struct Groups_List
       with locations #size Group* long and count_groups set to 0
    """

    cdef Groups_List* list_new

    list_new              = <Groups_List *>malloc( sizeof( Groups_List ) )
    if not list_new:
        raise MemoryError()

    list_new.board_groups = <Group **>malloc( size * sizeof( Group* ) )
    if not list_new.board_groups:
        raise MemoryError()

    list_new.count_groups = 0

    return list_new


@cython.boundscheck( False )
@cython.wraparound(  False )
cdef void groups_list_add( Group* group, Groups_List* groups_list ):
    """
       add group to list and increment groups count
    """

    groups_list.board_groups[ groups_list.count_groups ] = group
    groups_list.count_groups += 1


@cython.boundscheck( False )
@cython.wraparound(  False )
cdef void groups_list_add_unique( Group* group, Groups_List* groups_list ):
    """
       check if a group is already in the list, return if so
       add group to list if not
    """

    cdef int i

    # loop over array
    for i in range( groups_list.count_groups ):

        # check if group is present
        if group == groups_list.board_groups[ i ]:

            # group is present, return
            return

    # group is not present, add to group
    groups_list.board_groups[ groups_list.count_groups ] = group
    groups_list.count_groups += 1


@cython.boundscheck( False )
@cython.wraparound(  False )
cdef void groups_list_remove( Group* group, Groups_List* groups_list ):
    """
       remove group from list and decrement groups count
    """

    cdef int i

    # loop over array
    for i in range( groups_list.count_groups ):

        # check if group is present
        if groups_list.board_groups[ i ] == group:

            # group is present, move last group to this location
            # and decrement groups count
            groups_list.count_groups -= 1
            groups_list.board_groups[ i ] = groups_list.board_groups[ groups_list.count_groups ]
            return

    # TODO this should not happen, create error for this??
    print( "Group not found!!!!!!!!!!!!!!" )


############################################################################
#   Locations_List functions                                               #
#                                                                          #
############################################################################


@cython.boundscheck( False )
@cython.wraparound(  False )
cdef Locations_List* locations_list_new( short size ):
    """
       create new struct Locations_List
       with locations #size short long and count set to 0
    """

    cdef Locations_List* list_new

    # allocate memory for Group
    list_new           = <Locations_List *>malloc( sizeof( Locations_List ) )
    if not list_new:
        raise MemoryError()

    # allocate memory for locations
    list_new.locations = <short *>malloc( size * sizeof( short ) )
    if not list_new.locations:
        raise MemoryError()

    # set count to 0
    list_new.count     = 0

    # set size
    list_new.size      = size

    return list_new

@cython.boundscheck( False )
@cython.wraparound(  False )
cdef void locations_list_destroy( Locations_List* locations_list ):
    """
       free memory location of locations_list and locations
    """

    # check if locations_list exists
    if locations_list is not NULL:

        # check if locations exists
        if locations_list.locations is not NULL:

            # free locations
            free( locations_list.locations )

        # free locations_list
        free( locations_list )

@cython.boundscheck( False )
@cython.wraparound(  False )
cdef void locations_list_remove_location( Locations_List* locations_list, short location ):
    """
       remove location from list
    """

    cdef int i

    # loop over array
    for i in range( locations_list.count ):

        # check if [ i ] == location
        if locations_list.locations[ i ] == location:

            # location found, move last value to this location
            # and decrement count
            locations_list.count -= 1
            locations_list.locations[ i ] = locations_list.locations[ locations_list.count ]
            return

    # TODO this should not happen, create error for this??
    print( "location not found!!!!!!!!!!!!!!" )


@cython.boundscheck( False )
@cython.wraparound(  False )
cdef void locations_list_add_location( Locations_List* locations_list, short location ):
    """
       add location to list and increment count
    """

    locations_list.locations[ locations_list.count ] = location
    locations_list.count += 1


@cython.boundscheck( False )
@cython.wraparound(  False )
cdef void locations_list_add_location_increment( Locations_List* locations_list, short location ):
    """
       check if list can hold one more location, resize list if not
       add location to list and increment count
    """

    if locations_list.count == locations_list.size:

        locations_list.size += 10
        locations_list.locations = <short *>realloc( locations_list.locations, locations_list.size * sizeof( short ) )
        if not locations_list.locations:
            print("MEM ERROR")
            raise MemoryError()


    locations_list.locations[ locations_list.count ] = location
    locations_list.count += 1


@cython.boundscheck( False )
@cython.wraparound(  False )
@cython.nonecheck(   False )
cdef void locations_list_add_location_unique( Locations_List* locations_list, short location ):
    """
       check if location is present in list, return if so
       add location to list if not
    """

    cdef int i

    # loop over array
    for i in range( locations_list.count ):

        # check if location is present
        if location == locations_list.locations[ i ]:

            # location found, do nothing -> return
            return

    # add location to list and increment count
    locations_list.locations[ locations_list.count ] = location
    locations_list.count += 1


############################################################################
#   neighbor creation functions                                            #
#                                                                          #
############################################################################


@cython.boundscheck( False )
@cython.wraparound(  False )
cdef short calculate_board_location( char x, char y, char size ):
    """
       return location on board
       no checks on outside board
       x = columns
       y = rows           
    """

    # return board location
    return x + ( y * size )


@cython.boundscheck( False )
@cython.wraparound(  False )
cdef short calculate_board_location_or_border( char x, char y, char size ):
    """
       return location on board or borderlocation
       board locations = [ 0, size * size )
       border location = size * size
       x = columns
       y = rows
    """

    # check if x or y are outside board
    if x < 0 or y < 0 or x >= size or y >= size:

        # return border location
        return size * size

    # return board location
    return calculate_board_location( x, y, size )


cdef short* get_neighbors( char size ):
    """
       create array for every board location with all 4 direct neighbor locations
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
    cdef short* neighbor = <short *>malloc( size * size * 4 * sizeof( short ) )
    if not neighbor:
        raise MemoryError()

    cdef short location
    cdef char x, y

    # add all direct neighbors to every board location
    for y in range( size ):

        for x in range( size ):

            location = ( x + ( y * size ) ) * 4
            neighbor[ location + 0 ] = calculate_board_location_or_border( x - 1, y    , size )
            neighbor[ location + 1 ] = calculate_board_location_or_border( x + 1, y    , size )
            neighbor[ location + 2 ] = calculate_board_location_or_border( x    , y - 1, size )
            neighbor[ location + 3 ] = calculate_board_location_or_border( x    , y + 1, size )

    return neighbor

@cython.boundscheck( False )
@cython.wraparound(  False )
cdef short* get_3x3_neighbors( char size ):
    """
       create for every board location array with all 8 surrounding neighbor locations
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
    cdef short* neighbor3x3 = <short *>malloc( size * size * 8 * sizeof( short ) )
    if not neighbor3x3:
        raise MemoryError()

    cdef short location
    cdef char x, y

    # add all surrounding neighbors to every board location
    for x in range( size ):

        for y in range( size ):

            location = ( x + ( y * size ) ) * 8
            neighbor3x3[ location + 0 ] = calculate_board_location_or_border( x    , y - 1, size )
            neighbor3x3[ location + 1 ] = calculate_board_location_or_border( x - 1, y    , size )
            neighbor3x3[ location + 2 ] = calculate_board_location_or_border( x + 1, y    , size )
            neighbor3x3[ location + 3 ] = calculate_board_location_or_border( x    , y + 1, size )

            neighbor3x3[ location + 4 ] = calculate_board_location_or_border( x - 1, y - 1, size )
            neighbor3x3[ location + 5 ] = calculate_board_location_or_border( x + 1, y - 1, size )
            neighbor3x3[ location + 6 ] = calculate_board_location_or_border( x - 1, y + 1, size )
            neighbor3x3[ location + 7 ] = calculate_board_location_or_border( x + 1, y + 1, size )

    return neighbor3x3

@cython.boundscheck( False )
@cython.wraparound(  False )
cdef short* get_12d_neighbors( char size ):
    """
       create array for every board location with 12d star neighbor locations
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
    cdef short* neighbor12d = <short *>malloc( size * size * 12 * sizeof( short ) )
    if not neighbor12d:
        raise MemoryError()

    cdef short location
    cdef char x, y

    # add all 12d neighbors to every board location
    for x in range( size ):

        for y in range( size ):

            location = ( x + ( y * size ) ) * 12
            neighbor12d[ location +  4 ] = calculate_board_location_or_border( x    , y - 2, size )

            neighbor12d[ location +  1 ] = calculate_board_location_or_border( x - 1, y - 1, size )
            neighbor12d[ location +  5 ] = calculate_board_location_or_border( x    , y - 1, size )
            neighbor12d[ location +  8 ] = calculate_board_location_or_border( x + 1, y - 1, size )

            neighbor12d[ location +  0 ] = calculate_board_location_or_border( x - 2, y    , size )
            neighbor12d[ location +  2 ] = calculate_board_location_or_border( x - 1, y    , size )
            neighbor12d[ location +  9 ] = calculate_board_location_or_border( x + 1, y    , size )
            neighbor12d[ location + 11 ] = calculate_board_location_or_border( x + 2, y    , size )

            neighbor12d[ location +  3 ] = calculate_board_location_or_border( x - 1, y + 1, size )
            neighbor12d[ location +  6 ] = calculate_board_location_or_border( x    , y + 1, size )
            neighbor12d[ location + 10 ] = calculate_board_location_or_border( x + 1, y + 1, size )

            neighbor12d[ location +  7 ] = calculate_board_location_or_border( x    , y + 2, size )

    return neighbor12d