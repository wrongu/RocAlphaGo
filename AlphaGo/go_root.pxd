from AlphaGo.go cimport GameState
from AlphaGo.go_data cimport _BORDER, _EMPTY, _BLACK, _WHITE, _PASS, Group, Groups_List, Locations_List, group_new

cdef class RootState:

    ############################################################################
    #   variables declarations                                                 #
    #                                                                          #
    ############################################################################

    cdef short  size
    cdef short  board_size

    # empty group
    cdef Group *group_empty

    # border group
    cdef Group *group_border

    # arrays, neighbor arrays pointers
    cdef short *neighbor
    cdef short *neighbor3x3
    cdef short *neighbor12d

    ############################################################################
    #   cdef init functions                                                    #
    #                                                                          #
    ############################################################################

    cdef short  calculate_board_location(           self, char x, char y )
    cdef short  calculate_board_location_or_border( self, char x, char y )

    cdef void   set_neighbors(        self, int size )
    cdef void   set_3x3_neighbors(    self, int size )
    cdef void   set_12d_neighbors(    self, int size )

    ############################################################################
    #   public functions ( c only )                                            #
    #                                                                          #
    ############################################################################

    cdef GameState get_root_state( self )

    ############################################################################
    #   public functions                                                       #
    #                                                                          #
    ############################################################################

    # def get_root_game_state( self )