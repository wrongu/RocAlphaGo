############################################################################
#   Global constants                                                       #
#                                                                          #
############################################################################

cdef enum action_t:
    PASS = -1
    PLAY = +1


# observe: stones > _EMPTY
#          border < _EMPTY
# be aware you should NOT use != _EMPTY as this includes border locations
cdef enum stone_t:
    BORDER = 1
    EMPTY = 2
    WHITE = 3
    BLACK = 4


# Used for group stone, liberty locations, legal move and sensible move in feature processing.
cdef enum group_t:
    STONE = 0
    LIBERTY = 1
    CAPTURE = 2
    FREE = 3
    LEGAL = 4
    EYE = 5
