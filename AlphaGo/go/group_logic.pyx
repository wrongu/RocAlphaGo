# cython: wraparound=False
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
from cython.operator cimport dereference as d


############################################################################
#   Simple & fast group functions                                          #
#                                                                          #
############################################################################

cdef group_ptr_t group_new(stone_t color):
    """Create new struct Group with empty set of locations.
    """

    # Initialize group and its members using C++ "new". Members are automatically initialized to
    # defaults.
    cdef group_ptr_t group = group_ptr_t(new Group())
    d(group).color = color

    return group

cdef group_ptr_t group_duplicate(group_ptr_t group):
    """Create a (deep) copy of the given group-pointer.
    """

    # Create new Group object with same color.
    cdef group_ptr_t new_group = group_new(d(group).color)

    # Copy each field individually; using C++ this amounts to a 'deep copy' of group.locations.
    d(new_group).count_stones = d(group).count_stones
    d(new_group).count_liberty = d(group).count_liberty
    d(new_group).locations = d(group).locations

    return new_group

cdef void group_add_stone(group_ptr_t group, location_t location):
    """Update location as STONE and update counts.
    """

    cdef group_t current_type = group_lookup(group, location)

    # Update counts
    if current_type != group_t.STONE:
        # Note: '+=' does not work with the d() operator. cython bug!
        d(group).count_stones = d(group).count_stones + 1
        if current_type == group_t.LIBERTY:
            d(group).count_liberty = d(group).count_liberty - 1

    # Flag location as STONE
    d(group).locations[location] = group_t.STONE

cdef void group_remove_stone(group_ptr_t group, location_t location):
    """Update location as FREE and update counts.
    """

    cdef group_t current_type = group_lookup(group, location)

    # Check if a stone is present
    if current_type == group_t.STONE:
        # Stone present, decrement stone count and clear location
        d(group).count_stones = d(group).count_stones - 1
        d(group).locations.erase(location)

cdef void group_merge(group_ptr_t group, group_ptr_t other):
    """Merge groups by copying stones from 'other' into 'group'. Ensures that liberties are
       appropriately updated. Leaves 'other' unchanged.
    """
    cdef location_t loc
    cdef group_t val
    for loc, val in d(other).locations:
        # Add all stones from 'other' into 'group'
        if val == group_t.STONE:
            group_add_stone(group, loc)
        # Only add liberties from 'other' into 'group' if it would not overwrite something else
        elif val == group_t.LIBERTY and group_lookup(group, loc) == group_t.FREE:
            group_add_liberty(group, loc)

cdef location_t group_get_stone(group_ptr_t group):
    """Return any one location where there is a group_t.STONE or -1 if not found.
    """

    # Iterate until a STONE is found.
    cdef location_t loc
    cdef group_t val
    for loc, val in d(group).locations:
        if val == group_t.STONE:
            return loc
    return -1

cdef void group_add_liberty(group_ptr_t group, location_t location):
    """Update location as LIBERTY and update counts.
    """

    cdef group_t current_type = group_lookup(group, location)

    # Update counts
    if current_type != group_t.LIBERTY:
        d(group).count_liberty = d(group).count_liberty + 1
        if current_type == group_t.STONE:
            d(group).count_stones = d(group).count_stones - 1

    # Set LIBERTY
    d(group).locations[location] = group_t.LIBERTY

cdef void group_remove_liberty(group_ptr_t group, location_t location):
    """Update location as FREE and update counts.
    """

    cdef group_t current_type = group_lookup(group, location)

    if current_type == group_t.LIBERTY:
        # Decrement liberty count and clear location
        d(group).count_liberty = d(group).count_liberty - 1
        d(group).locations.erase(location)

cdef location_t group_get_liberty(group_ptr_t group):
    """Return any one location that is flagged as LIBERTY or -1 if not found.
    """

    # Iterate until a LIBERTY is found.
    cdef location_t loc
    cdef group_t val
    for loc, val in d(group).locations:
        if val == group_t.LIBERTY:
            return loc
    return -1

cdef group_t group_lookup(group_ptr_t group, location_t location):
    """Return stone type at 'location' or FREE.
    """

    # Look up location in map using C++ iterator comparison (i.e. 'location in group.locations')
    if d(group).locations.find(location) != d(group).locations.end():
        return d(group).locations[location]
    else:
        return group_t.FREE
