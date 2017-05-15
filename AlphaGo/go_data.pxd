
############################################################################
#   constants                                                              #
#                                                                          #
############################################################################

# TODO find out if these are really used as compile time-constant

# value for PASS move
cdef char _PASS

# observe: stones > EMPTY
#          border < EMPTY
# be aware you should NOT use != EMPTY as this includes border locations
cdef char _BORDER
cdef char _EMPTY
cdef char _WHITE
cdef char _BLACK

# used for group stone and liberty locations
cdef char _FREE
cdef char _STONE
cdef char _LIBERTY
cdef char _CAPTURE

# value used to generate pattern hashes
cdef char _HASHVALUE


############################################################################
#   Structs                                                                #
#                                                                          #
############################################################################


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


############################################################################
#   group functions                                                        #
#                                                                          #
############################################################################

cdef Group* group_new(              char colour,  short size )
cdef Group* group_duplicate(        Group* group, short size )
cdef void   group_destroy(          Group* group )

cdef void   group_add_stone(        Group* group, short location )
cdef void   group_remove_stone(     Group* group, short location )
cdef short  group_location_stone(   Group* group, short size )

cdef void   group_add_liberty(      Group* group, short location )
cdef void   group_remove_liberty(   Group* group, short location )
cdef short  group_location_liberty( Group* group, short size )

############################################################################
#   Groups_List functions                                                  #
#                                                                          #
############################################################################

cdef void   groups_list_add(        Group* group, Groups_List* groups_list )
cdef void   groups_list_add_unique( Group* group, Groups_List* groups_list )
cdef void   groups_list_remove(     Group* group, Groups_List* groups_list )

############################################################################
#   Locations_List functions                                               #
#                                                                          #
############################################################################

cdef Locations_List* locations_list_new(      short size )
cdef void locations_list_destroy(             Locations_List* locations_list )
cdef void locations_list_remove_location(     Locations_List* locations_list, short location )
cdef void locations_list_add_location(        Locations_List* locations_list, short location )
cdef void locations_list_add_location_unique( Locations_List* locations_list, short location )
