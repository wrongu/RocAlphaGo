from AlphaGo.go.constants cimport stone_t, group_t
from libcpp.unordered_map cimport unordered_map
from libcpp.memory cimport shared_ptr
from libcpp.vector cimport vector


############################################################################
#   Struct definition                                                      #
#                                                                          #
############################################################################

ctypedef short location_t

"""Class to store stone and liberty locations of a single 'group'

   'locations' is a map from 1d board location to a 'group_t' type. Most groups only store
   STONE and LIBERTY data. Other enum values used by ladder checks.

   Both count_stones and count_liberty are tracked as stones are added and removed from the group.

   Board size should be available from the context. This is relevant for modulo-arithmetic involved
   in computing locations.

   Note on use of cppclass as opposed to 'structs' or 'extension types':
   - The cppclass is treated as a plain C++ object, allowing smart pointers, vectors, etc.
   - Unlike 'structs', the cppclass has C++ semantics, with a constructor, copy constructor, and
     destructor. Copies are 'deep' automatically.
"""
cdef cppclass Group:
    stone_t color
    short count_stones, count_liberty
    unordered_map[location_t, group_t] locations

ctypedef shared_ptr[Group] group_ptr_t  # smart pointer with reference counting wrapping a 'Group'
ctypedef vector[group_ptr_t] board_group_t  # type for group-lookup by board position


############################################################################
#   Simple & fast group functions                                          #
#                                                                          #
############################################################################

cdef group_ptr_t group_new(stone_t color)
"""Create new Group with empty set of locations.
"""

cdef group_ptr_t group_duplicate(group_ptr_t group)
"""Create a (deep) copy of the given group-pointer.
"""

cdef void group_add_stone(group_ptr_t group, location_t location)
"""Update location as STONE and update counts.
"""

cdef void group_remove_stone(group_ptr_t group, location_t location)
"""Update location as FREE and update counts.
"""

cdef void group_merge(group_ptr_t group, group_ptr_t other)
"""Merge groups by copying stones from 'other' into 'group'. Ensures that liberties are
   appropriately updated. Leaves 'other' unchanged.
"""

cdef location_t group_get_stone(group_ptr_t group)
"""Return any one location where there is a STONE.
"""

cdef void group_add_liberty(group_ptr_t group, location_t location)
"""Update location as LIBERTY and update counts.
"""

cdef void group_remove_liberty(group_ptr_t group, location_t location)
"""Update location as FREE and update counts.
"""

cdef location_t group_get_liberty(group_ptr_t group)
"""Return any one location that is flagged as LIBERTY.
"""

cdef group_t group_lookup(group_ptr_t group, location_t location)
"""Return stone type at 'location' or FREE.
"""
