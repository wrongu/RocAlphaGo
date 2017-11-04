import numpy as np
cimport numpy as np

############################################################################
#   Global constants                                                       #
#                                                                          #
############################################################################

# Note that "enum" is the best way to create a compile-time constant in cython.

cpdef enum GAME:
    # value for _PASS move
    _PASS = -1

cpdef enum STONES:
    # observe: stones > _EMPTY
    #          border < _EMPTY
    # be aware you should NOT use != _EMPTY as this includes border locations
    _BORDER = 1
    _EMPTY = 2
    _WHITE = 3
    _BLACK = 4

cpdef enum PATTERNS:
    # value used to generate pattern hashes
    _HASHVALUE = 33

cpdef enum GROUP:
    # used for group stone, liberty locations, legal move and sensible move
    _FREE = 3
    _STONE = 0
    _LIBERTY = 1
    _CAPTURE = 2
    _LEGAL = 4
    _EYE = 5


############################################################################
#   Structs                                                                #
#                                                                          #
############################################################################

# Notes on use of structs over 'extension types':
#
# A struct has the advantage of being completely C, no python wrapper so no python overhead.
#
# compared to a cdef class a struct has some advantages:
# - C only, no python overhead
# - able to get a pointer to it
# - smaller in size
#
# drawbacks
# - have to be Malloc created and freed after use -> memory leak
# - no convenient functions available
# - no boundchecks

"""
   struct to store group stone and liberty locations

   locations is a char pointer array of size board_size and initialized
   to _FREE. after adding a stone/liberty that location is set to
   _STONE/_LIBERTY and count_stones/count_liberty is incremented

   note that a stone location can never be a liberty location,
   if a stone is placed on a liberty location liberty_count is decremented

   it works as a dictionary so lookup time for a location is O(1)
   looping over all stone/liberty location could be optimized by adding
   two lists containing stone/liberty locations

   TODO check if this dictionary implementation is faster on average
   use as a two list implementation
"""
cdef struct Group:
    char* locations
    short count_stones
    short count_liberty
    char  colour

"""
   struct to store a list of Group

   board_groups is a Group pointer array of size #size and containing
   #count_groups groups

   TODO convert to c++ list?
"""
cdef struct Groups_List:
    Group** board_groups
    short   count_groups
    short   size

"""
   struct to store a list of short (board locations)

   locations is a short pointer array of size #size and containing
   #count locations

   TODO convert to c++ list and/or set
"""
cdef struct Locations_List:
    short* locations
    short  count
    short  size


############################################################################
#   group functions                                                        #
#                                                                          #
############################################################################

cdef Group* group_new(char colour, short size)
"""
   create new struct Group
   with locations #size char long initialized to _FREE
"""

cdef Group* group_duplicate(Group* group, short size)
"""
   create new struct Group initialized as a duplicate of group
"""

cdef void group_destroy(Group* group)
"""
   free memory location of group and locations
"""

cdef void group_add_stone(Group* group, short location)
"""
   update location as _STONE
   update liberty count if it was a liberty location

   n.b. stone count is not incremented if a stone was present already
"""

cdef void group_remove_stone(Group* group, short location)
"""
   update location as _FREE
   update stone count if it was a stone location
"""

cdef short group_location_stone(Group* group, short size)
"""
   return first location where a _STONE is located
"""

cdef void group_add_liberty(Group* group, short location)
"""
   update location as _LIBERTY
   update liberty count if it was a _FREE location

   n.b. liberty count is not incremented if a stone was present already
"""

cdef void group_remove_liberty(Group* group, short location)
"""
   update location as _FREE
   update liberty count if it was a _LIBERTY location

   n.b. liberty count is not decremented if location is a _FREE location
"""

cdef short group_location_liberty(Group* group, short size)
"""
   return location where a _LIBERTY is located
"""

############################################################################
#   Groups_List functions                                                  #
#                                                                          #
############################################################################

cdef Groups_List* groups_list_new(short size)
"""
   create new struct Groups_List
   with locations #size Group* long and count_groups set to 0
"""

cdef void groups_list_add(Group* group, Groups_List* groups_list)
"""
   add group to list and increment groups count
"""

cdef void groups_list_add_unique(Group* group, Groups_List* groups_list)
"""
   check if a group is already in the list, return if so
   add group to list if not
"""

cdef void groups_list_remove(Group* group, Groups_List* groups_list)
"""
   remove group from list and decrement groups count
"""

############################################################################
#   Locations_List functions                                               #
#                                                                          #
############################################################################

cdef Locations_List* locations_list_new(short size)
"""
   create new struct Locations_List
   with locations #size short long and count set to 0
"""

cdef void locations_list_destroy(Locations_List* locations_list)
"""
   free memory location of locations_list and locations
"""

cdef void locations_list_remove_location(Locations_List* locations_list, short location)
"""
   remove location from list
"""

cdef void locations_list_add_location(Locations_List* locations_list, short location)
"""
   add location to list and increment count
"""

cdef void locations_list_add_location_increment(Locations_List* locations_list, short location)
"""
   check if list can hold one more location, resize list if not
   add location to list and increment count
"""

cdef void locations_list_add_location_unique(Locations_List* locations_list, short location)
"""
   check if location is present in list, return if so
   add location to list if not
"""

############################################################################
#   neighbor creation functions                                            #
#                                                                          #
############################################################################

cdef short calculate_board_location(char x, char y, char size)
"""
   return location on board
   no checks on outside board
   x = columns
   y = rows
"""

cdef short calculate_board_location_or_border(char x, char y, char size)
"""
   return location on board or borderlocation
   board locations = [ 0, size * size)
   border location = size * size
   x = columns
   y = rows
"""

cdef short* get_neighbors(char size)
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

cdef short* get_3x3_neighbors(char size)
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

cdef short* get_12d_neighbors(char size)
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

############################################################################
#   zobrist creation functions                                             #
#                                                                          #
############################################################################


cdef unsigned long long* get_zobrist_lookup(char size)
