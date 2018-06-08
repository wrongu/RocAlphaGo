from cython.operator cimport dereference as d


# Constant value used to generate pattern hashes
cdef int _HASHVALUE = 33


############################################################################
#   Helper functions to switch between types of coordinates                #
#                                                                          #
############################################################################

cdef location_t calculate_board_location(location_t x, location_t y, location_t size):
    """Return location on board
       no checks on outside board
       x = columns
       y = rows
    """

    # Return board location
    return x + (y * size)


cdef location_t calculate_board_location_or_border(location_t x, location_t y, location_t size):
    """Return location on board or borderlocation
       board locations = [0, size * size)
       border location = size * size
       x = columns
       y = rows
    """

    # Check if x or y are outside board
    if x < 0 or y < 0 or x >= size or y >= size:
        # Return border location
        return size * size

    # Return board location
    return calculate_board_location(x, y, size)


cdef tuple calculate_tuple_location(location_t location, location_t size):
    """1d index to 2d tuple location (x, y). Inverse of calculate_board_location()

       No sanity checks on bounds.
    """
    return divmod(location, size)

############################################################################
#   Neighbor/pattern lookup table creation functions                       #
#                                                                          #
############################################################################

cdef pattern_hash_t get_pattern_hash(board_group_t &board, location_t center, int pattern_size, pattern_t &pattern_lookup, int max_liberty=3):  # noqa:E501
    """Given a neighbor/pattern lookup table, computes a hash of the pattern around a location,
       treating each color + liberty combination as a unique value (up to max_liberty).
    """

    cdef int i
    cdef pattern_hash_t hsh = _HASHVALUE
    cdef group_ptr_t group

    # Index into neighbor12d array is 12x index into board, for example.
    center *= pattern_size

    # Hash color and liberty count of all locations in pattern around the center.
    for i in range(pattern_size):
        # Get group
        group = board[pattern_lookup[center + i]]

        # Hash color
        hsh += d(group).color
        hsh *= _HASHVALUE

        # Hash liberty count (up to max value)
        hsh += min(d(group).count_liberty, max_liberty)
        hsh *= _HASHVALUE

    return hsh

cdef pattern_t get_neighbors(location_t size):
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

    # Initialize empty vector
    cdef pattern_t neighbor = pattern_t(4 * size * size)

    cdef short location
    cdef location_t x, y

    # Add all direct neighbors to every board location
    for y in range(size):
        for x in range(size):
            location = 4 * calculate_board_location(x, y, size)
            neighbor[location + 0] = calculate_board_location_or_border(x - 1, y, size)
            neighbor[location + 1] = calculate_board_location_or_border(x + 1, y, size)
            neighbor[location + 2] = calculate_board_location_or_border(x, y - 1, size)
            neighbor[location + 3] = calculate_board_location_or_border(x, y + 1, size)

    return neighbor


cdef pattern_t get_3x3_neighbors(location_t size):
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

    # Initialize empty vector
    cdef pattern_t neighbor3x3 = pattern_t(8 * size * size)

    cdef short location
    cdef location_t x, y

    # Add all surrounding neighbors to every board location
    for x in range(size):
        for y in range(size):
            location = 8 * calculate_board_location(x, y, size)
            # Cardinal directions
            neighbor3x3[location + 0] = calculate_board_location_or_border(x, y - 1, size)
            neighbor3x3[location + 1] = calculate_board_location_or_border(x - 1, y, size)
            neighbor3x3[location + 2] = calculate_board_location_or_border(x + 1, y, size)
            neighbor3x3[location + 3] = calculate_board_location_or_border(x, y + 1, size)
            # Diagonal directions
            neighbor3x3[location + 4] = calculate_board_location_or_border(x - 1, y - 1, size)
            neighbor3x3[location + 5] = calculate_board_location_or_border(x + 1, y - 1, size)
            neighbor3x3[location + 6] = calculate_board_location_or_border(x - 1, y + 1, size)
            neighbor3x3[location + 7] = calculate_board_location_or_border(x + 1, y + 1, size)

    return neighbor3x3


cdef pattern_t get_12d_neighbors(location_t size):
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

    # Initialize empty vector
    cdef pattern_t neighbor12d = pattern_t(12 * size * size)

    cdef short location
    cdef location_t x, y

    # Add all 12d neighbors to every board location
    for x in range(size):
        for y in range(size):
            location = 12 * calculate_board_location(x, y, size)
            neighbor12d[location + 4] = calculate_board_location_or_border(x, y - 2, size)

            neighbor12d[location + 1] = calculate_board_location_or_border(x - 1, y - 1, size)
            neighbor12d[location + 5] = calculate_board_location_or_border(x, y - 1, size)
            neighbor12d[location + 8] = calculate_board_location_or_border(x + 1, y - 1, size)

            neighbor12d[location + 0] = calculate_board_location_or_border(x - 2, y, size)
            neighbor12d[location + 2] = calculate_board_location_or_border(x - 1, y, size)
            neighbor12d[location + 9] = calculate_board_location_or_border(x + 1, y, size)
            neighbor12d[location + 11] = calculate_board_location_or_border(x + 2, y, size)

            neighbor12d[location + 3] = calculate_board_location_or_border(x - 1, y + 1, size)
            neighbor12d[location + 6] = calculate_board_location_or_border(x, y + 1, size)
            neighbor12d[location + 10] = calculate_board_location_or_border(x + 1, y + 1, size)

            neighbor12d[location + 7] = calculate_board_location_or_border(x, y + 2, size)

    return neighbor12d
