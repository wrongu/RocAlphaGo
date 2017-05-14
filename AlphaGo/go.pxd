import numpy as np
cimport numpy as np
from AlphaGo.go_data cimport *


cdef class GameState:

    ############################################################################
    #   variables declarations                                                 #
    #                                                                          #
    ############################################################################

    # amount of locations on one side
    cdef char  size
    # amount of locations on board, size * size
    cdef short board_size

    # possible ko location
    cdef short ko                

    # list with all groups
    cdef Groups_List *groups_list
    # pointer to empty group
    cdef Group       *group_empty

    # list representing board locations as groups
    # a Group contains all group stone locations and group liberty locations
    cdef Group **board_groups

    cdef char player_current
    cdef char player_opponent

    cdef short capture_black
    cdef short capture_white

    cdef short passes_black
    cdef short passes_white

    # TODO replace list with struct
    cdef list history
    cdef Locations_List *moves_history
    # list with legal moves
    cdef Locations_List *moves_legal

    # array, keep track of 3x3 pattern hashes
    cdef long  *hash3x3

    # arrays, neighbor arrays pointers
    cdef short *neighbor
    cdef short *neighbor3x3
    cdef short *neighbor12d

    # zobrist
    cdef dict   hash_lookup
    cdef int    current_hash
    cdef set    previous_hashes

    ############################################################################
    #   init functions                                                         #
    #                                                                          #
    ############################################################################

    cdef void   initialize_duplicate( self, GameState copyState )

    ############################################################################
    #   private cdef functions used for game-play                              #
    #                                                                          #
    ############################################################################

    cdef bint   is_legal_move( self, short location, Group **board, short ko )
    cdef bint   has_liberty_after( self, short location, Group **board )
    cdef short  calculate_board_location( self, char x, char y )
    cdef tuple  calculate_tuple_location( self, short location )
    cdef void   set_moves_legal_list( self, Locations_List *moves_legal )

    cdef void   combine_groups( self, Group* group_keep, Group* group_remove, Group **board )
    cdef void   remove_group( self, Group* group_remove, Group **board, short* ko )
    cdef void   update_hashes( self, Group* group )
    cdef void   add_to_group( self, short location, Group **board, short* ko, short* count_captures )

    ############################################################################
    #   private cdef functions used for feature generation                     #
    #                                                                          #
    ############################################################################

    cdef long   generate_12d_hash( self, short centre )
    cdef long   generate_3x3_hash( self, short centre )
    cdef void   get_group_after( self, short* groups_after, char* locations, char* captures, short location )
    cdef bint   is_true_eye( self, short location, Locations_List* eyes, char owner )

    ############################################################################
    #   private cdef Ladder functions                                          #
    #                                                                          #
    ############################################################################

    cdef Groups_List* add_ladder_move(    self, short location, Group **board, short* ko )
    cdef void         unremove_group(     self, Group* group_remove, Group **board )
    cdef dict         get_capture_moves(  self, Group* group, char color, Group **board )
    cdef void         get_removed_groups( self, short location, Groups_List* removed_groups, Group **board, short* ko )
    cdef void         undo_ladder_move(   self, short location, Groups_List* removed_groups, short ko, Group **board, short* ko )
    cdef bint         is_ladder_escape_move(  self, Group **board, short* ko, short location_group, dict capture, short location, int maxDepth, char colour_group, char colour_chase )
    cdef bint         is_ladder_capture_move( self, Group **board, short* ko, short location_group, dict capture, short location, int maxDepth, char colour_group, char colour_chase )

    ############################################################################
    #   public cdef functions used by preprocessing                            #
    #                                                                          #
    ############################################################################

    cdef short*          get_groups_after(       self )
    cdef Locations_List* get_sensible_moves(     self )
    cdef list            get_neighbor_locations( self )
    cdef long            get_hash_12d(           self, short centre   )
    cdef long            get_hash_3x3(           self, short location )
    cdef char*           get_ladder_escapes(     self, int maxDepth   )
    cdef char*           get_ladder_captures(    self, int maxDepth   )
    cdef char            get_board_feature(      self, short location )

    ############################################################################
    #   public cdef functions used for game play                               #
    #                                                                          #
    ############################################################################

    cdef void         add_move(           self, short location )
    cdef GameState    new_state_add_move( self, short location )
    cdef char         get_winner_colour(  self, int komi )

    ############################################################################
    #   public def functions used for game play (Python)                       #
    #                                                                          #
    ############################################################################

    #def do_move( self, action )
    #def get_next_state( self, action )
    #def place_handicap( self, handicap )
    #def get_winner( self, char komi )
    #def get_legal_moves( self, include_eyes = True )
    #def is_legal( self, action )

    ############################################################################
    #   tests                                                                  #
    #                                                                          #
    ############################################################################

    cdef test( self )
    cdef test_cpp_fast( self )
