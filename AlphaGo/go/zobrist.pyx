# cython: initializedcheck=False
# cython: nonecheck=False
from cython.operator cimport dereference as d
import numpy as np
cimport numpy as np


cdef vector[zobrist_hash_t] get_zobrist_lookup(short size):
    """Generate zobrist lookup table for given board size (given as single-edge dimension)
    """

    # One entry per board location per color
    cdef int num_entries = size * size * 2
    cdef vector[zobrist_hash_t] table = vector[zobrist_hash_t](num_entries)

    # Initialize all zobrist hash lookup values
    for i in range(num_entries):
        table[i] = np.random.randint(np.iinfo(np.uint64).max, dtype='uint64')

    return table


cdef zobrist_hash_t update_hash_by_location(zobrist_hash_t current_hash, vector[zobrist_hash_t] table, location_t location, stone_t color):  # noqa: E501
    """Update zobrist hash for a single location and color. This applies to both adding and removing
       a stone.
    """

    # Even indices of the table used for white stones, odd used for black stones. Current hash is
    # XORed with the table entry for this combination of location and color.
    return current_hash ^ table[2 * location + <short>(color == stone_t.BLACK)]


cdef zobrist_hash_t update_hash_by_group(zobrist_hash_t current_hash, vector[zobrist_hash_t] table, group_ptr_t group):  # noqa: E501
    """Update zobrist hash for an entire group. This applies both to adding and removing groups.
    """

    cdef location_t loc
    cdef group_t value
    cdef stone_t color = d(group).color

    # Update the hash for every _STONE in this group
    for loc, value in d(group).locations:
        if value == group_t.STONE:
            current_hash = update_hash_by_location(current_hash, table, loc, color)

    return current_hash
