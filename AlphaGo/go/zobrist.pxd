from AlphaGo.go.constants cimport stone_t, group_t
from AlphaGo.go.group_logic cimport Group
from libcpp.vector cimport vector
from libcpp.memory cimport shared_ptr


ctypedef short location_t
ctypedef unsigned long long zobrist_hash_t
ctypedef shared_ptr[Group] group_ptr_t  # smart pointer with reference counting wrapping a 'Group'


cdef vector[zobrist_hash_t] get_zobrist_lookup(short size)
"""Generate zobrist lookup table for given board size (given as single-edge dimension)
"""


cdef zobrist_hash_t update_hash_by_location(zobrist_hash_t current_hash, vector[zobrist_hash_t] lut, location_t location, stone_t color)  # noqa: E501
"""Update zobrist hash for a single location and color.
"""


cdef zobrist_hash_t update_hash_by_group(zobrist_hash_t current_hash, vector[zobrist_hash_t] lut, group_ptr_t group)  # noqa: E501
"""Update zobrist hash for an entire group.
"""
