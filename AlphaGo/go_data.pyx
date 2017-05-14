cimport cython
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset, memchr

############################################################################
#   constants                                                              #
#                                                                          #
############################################################################


# value for PASS move
_PASS   = -1

# observe: stones > EMPTY
#          border < EMPTY
# be aware you should NOT use != EMPTY as this includes border locations
_BORDER = 1
_EMPTY  = 2
_WHITE  = 3
_BLACK  = 4

# used for group stone and liberty locations
_FREE    = 3
_STONE   = 0
_LIBERTY = 1
_CAPTURE = 2

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
       return location where a STONE is located
    """

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

    return <short>( <long>memchr(group.locations, _LIBERTY, size ) - <long>group.locations )


############################################################################
#   Groups_List functions                                                  #
#                                                                          #
############################################################################


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
