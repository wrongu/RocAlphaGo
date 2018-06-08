from AlphaGo.go.group_logic cimport Group
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr
import numpy as np
cimport numpy as np


############################################################################
#   Typedefs                                                               #
#                                                                          #
############################################################################

ctypedef short location_t
ctypedef unsigned long pattern_hash_t
ctypedef vector[location_t] pattern_t  # lookup of neighbor coordinates (or border)
ctypedef shared_ptr[Group] group_ptr_t  # smart pointer with reference counting wrapping a 'Group'
ctypedef vector[group_ptr_t] board_group_t  # type for group-lookup by board position


############################################################################
#   Helper functions to switch between types of coordinates                #
#                                                                          #
############################################################################

cdef location_t calculate_board_location(location_t x, location_t y, location_t size)
"""Return location on board
   no checks on outside board
   x = columns
   y = rows
"""

cdef location_t calculate_board_location_or_border(location_t x, location_t y, location_t size)
"""Return location on board or borderlocation
   board locations = [ 0, size * size)
   border location = size * size
   x = columns
   y = rows
"""

cdef tuple calculate_tuple_location(location_t index, location_t size)
"""1d index to 2d tuple location. Inverse of calculate_board_location()

   No sanity checks on bounds.
"""

############################################################################
#   Neighbor/pattern lookup table creation functions                       #
#                                                                          #
############################################################################

cdef pattern_hash_t get_pattern_hash(board_group_t &board, location_t center, int pattern_size, pattern_t &pattern_lookup, int max_liberty=*)  # noqa:E501
"""Given a neighbor/pattern lookup table, computes a hash of the pattern around a location,
   treating each color + liberty combination as a unique value (up to max_liberty).
"""

cdef pattern_t get_neighbors(location_t size)
"""Create array for every board location with all 4 direct neighbor locations
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

cdef pattern_t get_3x3_neighbors(location_t size)
"""Create for every board location array with all 8 surrounding neighbor locations
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

cdef pattern_t get_12d_neighbors(location_t size)
"""Create array for every board location with 12d star neighbor locations
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
