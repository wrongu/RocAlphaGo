import ast
import time
import numpy as np
cimport numpy as np
from numpy cimport ndarray
from libc.stdlib cimport malloc, free
from AlphaGo.go cimport GameState
from AlphaGo.go_data cimport _BLACK, _EMPTY, _STONE, _LIBERTY, _CAPTURE, _FREE, Group, Locations_List, locations_list_destroy

ctypedef char tensor_type
ctypedef int (*preprocess_method)( Preprocess, GameState, tensor_type[ :, ::1 ], short*, int )


cdef class Preprocess:

    ############################################################################
    #   variables declarations                                                 #
    #                                                                          #
    ############################################################################

    # all feature processors
    # TODO find correct type so an array can be used
    cdef preprocess_method *processors

    # list with all features used currently
    # TODO find correct type so an array can be used
    cdef list  feature_list

    # output tensor size
    cdef int   output_dim

    # board size
    cdef char  size
    cdef short board_size

    # pattern dictionaries
    cdef dict  pattern_nakade
    cdef dict  pattern_response_12d
    cdef dict  pattern_non_response_3x3

    # pattern dictionary sizes
    cdef int   pattern_nakade_size
    cdef int   pattern_response_12d_size
    cdef int   pattern_non_response_3x3_size

    ############################################################################
    #   Tensor generating functions                                            #
    #                                                                          #
    ############################################################################

    cdef int get_board(               self, GameState state, tensor_type[ :, ::1 ] tensor, short *groups_after, int offSet )
    cdef int get_turns_since(         self, GameState state, tensor_type[ :, ::1 ] tensor, short *groups_after, int offSet )
    cdef int get_liberties(           self, GameState state, tensor_type[ :, ::1 ] tensor, short *groups_after, int offSet )
    cdef int get_capture_size(        self, GameState state, tensor_type[ :, ::1 ] tensor, short *groups_after, int offSet )
    cdef int get_self_atari_size(     self, GameState state, tensor_type[ :, ::1 ] tensor, short *groups_after, int offSet )
    cdef int get_liberties_after(     self, GameState state, tensor_type[ :, ::1 ] tensor, short *groups_after, int offSet )
    cdef int get_ladder_capture(      self, GameState state, tensor_type[ :, ::1 ] tensor, short *groups_after, int offSet )
    cdef int get_ladder_escape(       self, GameState state, tensor_type[ :, ::1 ] tensor, short *groups_after, int offSet )
    cdef int get_sensibleness(        self, GameState state, tensor_type[ :, ::1 ] tensor, short *groups_after, int offSet )
    cdef int get_legal(               self, GameState state, tensor_type[ :, ::1 ] tensor, short *groups_after, int offSet )
    cdef int get_response(            self, GameState state, tensor_type[ :, ::1 ] tensor, short *groups_after, int offSet )
    cdef int get_save_atari(          self, GameState state, tensor_type[ :, ::1 ] tensor, short *groups_after, int offSet )
    cdef int get_neighbor(            self, GameState state, tensor_type[ :, ::1 ] tensor, short *groups_after, int offSet )
    cdef int get_nakade(              self, GameState state, tensor_type[ :, ::1 ] tensor, short *groups_after, int offSet )
    cdef int get_nakade_offset(       self, GameState state, tensor_type[ :, ::1 ] tensor, short *groups_after, int offSet )
    cdef int get_response_12d(        self, GameState state, tensor_type[ :, ::1 ] tensor, short *groups_after, int offSet )
    cdef int get_response_12d_offset( self, GameState state, tensor_type[ :, ::1 ] tensor, short *groups_after, int offSet )
    cdef int zeros(                   self, GameState state, tensor_type[ :, ::1 ] tensor, short *groups_after, int offSet )
    cdef int ones(                    self, GameState state, tensor_type[ :, ::1 ] tensor, short *groups_after, int offSet )
    cdef int colour(                  self, GameState state, tensor_type[ :, ::1 ] tensor, short *groups_after, int offSet )
    cdef int get_non_response_3x3(        self, GameState state, tensor_type[ :, ::1 ] tensor, short *groups_after, int offSet )
    cdef int get_non_response_3x3_offset( self, GameState state, tensor_type[ :, ::1 ] tensor, short *groups_after, int offSet )

    ############################################################################
    #   public cdef function                                                   #
    #                                                                          #
    ############################################################################

    cdef np.ndarray[ tensor_type, ndim=4 ] generate_tensor( self, GameState state )
