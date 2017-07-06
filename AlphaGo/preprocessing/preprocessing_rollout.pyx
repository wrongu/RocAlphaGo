# cython: profile=True
# cython: linetrace=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: initializedcheck=False
cimport cython
import numpy as np
cimport numpy as np


cdef class Preprocess:

    ############################################################################
    #   all variables are declared in the .pxd file                            #
    #                                                                          #
    ############################################################################


    """ -> variables, declared in preprocessing.pxd

    # all feature processors
    # TODO find correct type so an array can be used
    cdef list  processors

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

        -> variables, declared in preprocessing.pxd
    """


    ############################################################################
    #   Tensor generating functions                                            #
    #                                                                          #
    ############################################################################


    @cython.nonecheck(False)
    cdef int get_board(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet):
        """
           A feature encoding WHITE BLACK and EMPTY on separate planes.
           plane 0 always refers to the current player stones
           plane 1 to the opponent stones
           plane 2 to empty locations
        """

        cdef short  location
        cdef Group* group
        cdef int    plane
        cdef char   opponent = state.player_opponent

        # loop over all locations on board
        for location in range(self.board_size):

            group = state.board_groups[ location ]

            if group.colour == _EMPTY:

                plane = offSet + 2
            elif group.colour == opponent:

                plane = offSet + 1
            else:

                plane = offSet

            tensor[ plane, location ] = 1

        return offSet + 3


    @cython.nonecheck(False)
    cdef int get_turns_since(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet):
        """
           A feature encoding the age of the stone at each location up to 'maximum'

           Note:
           - the [maximum-1] plane is used for any stone with age greater than or equal to maximum
           - EMPTY locations are all-zero features
        """

        cdef short location
        cdef Locations_List *history = state.moves_history
        cdef int   age      = offSet + 7
        cdef dict  agesSet  = {}
        cdef int   i

        # set all stones to max age
        for i in range(history.count):

            location = history.locations[ i ]

            if location != _PASS and state.board_groups[ location ].colour > _EMPTY:

                tensor[ age, location ] = 1

        # start with newest stone
        i   = history.count - 1
        age = 0

        # loop over history backwards
        while age < 7 and i >= 0:

            location = history.locations[ i ]

            # if age has not been set yet
            if location != _PASS and not location in agesSet and state.board_groups[ location ].colour > _EMPTY:

                tensor[  offSet + age, location ] = 1
                tensor[  offSet + 7,   location ] = 0
                agesSet[ location ]               = location

            i   -= 1
            age += 1

        return offSet + 8


    @cython.nonecheck(False)
    cdef int get_liberties(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet):
        """
           A feature encoding the number of liberties of the group connected to the stone at
           each location

           Note:
           - there is no zero-liberties plane; the 0th plane indicates groups in atari
           - the [maximum-1] plane is used for any stone with liberties greater than or equal to maximum
           - EMPTY locations are all-zero features
        """

        cdef int    i, groupLiberty
        cdef Group* group
        cdef short  location

        for location in range(self.board_size):

            group = state.board_groups[ location ]

            if group.colour > _EMPTY:

                groupLiberty = group.count_liberty - 1

                # check max liberty count
                if groupLiberty > 7:

                    groupLiberty = 7

                groupLiberty += offSet

                tensor[ groupLiberty, location ] = 1

        return offSet + 8


    @cython.nonecheck(False)
    cdef int get_ladder_capture(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet):
        """
           A feature wrapping GameState.is_ladder_capture().
           check if an opponent group can be captured in a ladder
        """

        cdef int   location
        cdef char* captures = state.get_ladder_captures(80)

        # loop over all groups on board
        for location in range(state.board_size):

            if captures[ location ] != _FREE:

                tensor[ offSet, location ] = 1

        # free captures
        free(captures)

        return offSet + 1


    @cython.nonecheck(False)
    cdef int get_ladder_escape(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet):
        """
           A feature wrapping GameState.is_ladder_escape().
           check if player_current group can escape ladder
        """

        cdef int   location
        cdef char* escapes = state.get_ladder_escapes(80)

        # loop over all groups on board
        for location in range(state.board_size):

            if escapes[ location ] != _FREE:

                tensor[ offSet, location ] = 1

        # free escapes
        free(escapes)

        return offSet + 1


    @cython.nonecheck(False)
    cdef int get_sensibleness(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet):
        """
           A move is 'sensible' if it is legal and if it does not fill the current_player's own eye
        """

        cdef int    i
        cdef short  location
        cdef Group* group

        # set all legal moves to 1
        for i in range(state.moves_legal.count):

            tensor[ offSet, state.moves_legal.locations[ i ] ] = 1

        # list can increment but a big enough starting value is important
        cdef Locations_List* eyes  = locations_list_new(15)

        # loop over all board groups
        for i in range(state.groups_list.count_groups):

            group = state.groups_list.board_groups[ i ]

            # if group is current player
            if group.colour == state.player_current:

                # loop over liberties because they are possible eyes
                for location in range(self.board_size):

                    # check liberty location as possible eye
                    if group.locations[ location ] == _LIBERTY:

                        # check if location is an eye
                        if state.is_true_eye(location, eyes, state.player_current):

                            tensor[ offSet, location ] = 0

        locations_list_destroy(eyes)

        return offSet + 1


    @cython.nonecheck(False)
    cdef int get_legal(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet):
        """
           Zero at all illegal moves, one at all legal moves. Unlike sensibleness, no eye check is done
           not used??
        """

        cdef short location

        # loop over all legal moves and set to one
        for location in range(state.moves_legal.count):

            tensor[ offSet, state.moves_legal.locations[ location ] ] = 1

        return offSet + 1


    @cython.nonecheck(False)
    cdef int get_response(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet):
        """
           single feature plane encoding whether this location matches any of the response
           patterns, for now it only checks the 12d response patterns as we do not use the
           3x3 response patterns.

           TODO 
           - decide if we consider nakade patterns response patterns as well
           - optimization? 12d response patterns are calculated twice..          
        """

        cdef short location, location_x, location_y, last_move, last_move_x, last_move_y
        cdef int   i, plane, id
        cdef long  hash_base, hash_pattern
        cdef short *neighbor12d = state.neighbor12d

        # get last move
        last_move = state.moves_history.locations[ state.moves_history.count - 1 ]

        # check if last move is not _PASS
        if last_move != _PASS:

            # get 12d pattern hash of last move location and colour
            hash_base = state.get_hash_12d(last_move)

            # calculate last_move x and y
            last_move_x = last_move / state.size
            last_move_y = last_move % state.size

            # last_move location in neighbor12d array
            last_move *= 12

            # loop over all locations in 12d shape
            for i in range(12):

                # get location
                location = neighbor12d[last_move + i]

                # check if location is empty
                if state.board_groups[ location ].colour == _EMPTY:

                    # calculate location x and y
                    location_x = (location / state.size) - last_move_x
                    location_y = (location % state.size) - last_move_y
                
                    # calculate 12d response pattern hash
                    hash_pattern  = hash_base + location_x
                    hash_pattern *= _HASHVALUE
                    hash_pattern += location_y

                    # dictionary lookup
                    id = self.pattern_response_12d.get( hash_pattern )

                    if id >= 0:

                        tensor[ offSet, location ] = 1

        return offSet + 1


    @cython.nonecheck(False)
    cdef int get_save_atari(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet):
        """
           A feature wrapping GameState.is_ladder_escape().
           check if player_current group can escape atari for at least one turn
        """

        cdef int   location
        cdef char* escapes = state.get_ladder_escapes(1)

        # loop over all groups on board
        for location in range(state.board_size):

            if escapes[ location ] != _FREE:

                tensor[ offSet, location ] = 1

        # free escapes
        free(escapes)

        return offSet + 1


    @cython.nonecheck(False)
    cdef int get_neighbor(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet):
        """
           encode last move neighbor positions in two planes:
           - horizontal & vertical / direct neighbor
           - diagonal neighbor
        """

        cdef short location, last_move
        cdef int   i, plane
        cdef short *neighbor3x3 = state.neighbor3x3

        # get last move
        last_move = state.moves_history.locations[ state.moves_history.count - 1 ]

        # check if last move is not _PASS
        if last_move != _PASS:

            # last_move location in neighbor3x3 array
            last_move *= 8

            # direct neighbor plane is plane offset
            plane = offSet

            # loop over direct neighbor
            # 0,1,2,3 are direct neighbor locations
            for i in range(4):

                # get neighbor location
                location = neighbor3x3[ last_move + i ]

                # check if location is empty
                if state.board_groups[ location ].colour == _EMPTY:

                    tensor[ plane, location ] = 1

            # diagonal neighbor plane is plane offset + 1
            plane = offSet + 1

            # loop over diagonal neighbor
            # 4,5,6,7 are diagonal neighbor locations
            for i in range(4, 8):

                # get neighbor location
                location = neighbor3x3[ last_move + i ]

                # check if location is empty
                if state.board_groups[ location ].colour == _EMPTY:

                    tensor[ plane, location ] = 1

        return offSet + 2


    @cython.nonecheck(False)
    cdef int get_nakade(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet):
        """
           A nakade pattern is a 12d pattern on a location a stone was captured before
           it is unclear if a max size of the captured group has to be considered and
           how recent the capture event should have been

           the 12d pattern can be encoded without stone colour and liberty count
           unclear if a border location should be considered a stone or liberty

           pattern lookup value is being set instead of 1
        """

        # TODO tensor type has to be float

        return offSet + 1


    @cython.nonecheck(False)
    cdef int get_nakade_offset(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet):
        """
           A nakade pattern is a 12d pattern on a location a stone was captured before
           it is unclear if a max size of the captured group has to be considered and
           how recent the capture event should have been

           the 12d pattern can be encoded without stone colour and liberty count
           unclear if a border location should be considered a stone or liberty

           #pattern_id is offset
        """

        return offSet + self.pattern_nakade_size


    @cython.nonecheck(False)
    cdef int get_response_12d(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet):
        """
           Set 12d hash pattern for 12d shape around last move
           pattern lookup value is being set instead of 1
        """

        # get last move location
        # check for pass

        return offSet + 1


    @cython.nonecheck(False)
    cdef int get_response_12d_offset(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet):
        """
           Check all empty locations in a 12d shape around the last move for being a 12d response
           pattern match
           #pattern_id is offset

           base hash is 12d pattern hash of last move location + colour
           add relative position of every empty location in a 12d shape to get 12d response pattern hash

             c                        hash                    x   y
            ...       location a has: state.get_hash_12d(x), -1,  0
           .ax..      location b has: state.get_hash_12d(x), +1, -1                 
            ..b       location c has: state.get_hash_12d(x),  0, +2  
             .

           12d response pattern hash value is calculated by:
           ( ( hash + x ) * _HASHVALUE ) + y
        """

        cdef short location, location_x, location_y, last_move, last_move_x, last_move_y
        cdef int   i, plane, id
        cdef long  hash_base, hash_pattern
        cdef short *neighbor12d = state.neighbor12d

        # get last move
        last_move = state.moves_history.locations[ state.moves_history.count - 1 ]

        # check if last move is not _PASS
        if last_move != _PASS:

            # get 12d pattern hash of last move location and colour
            hash_base = state.get_hash_12d(last_move)

            # calculate last_move x and y
            last_move_x = last_move / state.size
            last_move_y = last_move % state.size

            # last_move location in neighbor12d array
            last_move *= 12

            # loop over all locations in 12d shape
            for i in range(12):

                # get location
                location = neighbor12d[last_move + i]

                # check if location is empty
                if state.board_groups[ location ].colour == _EMPTY:

                    # calculate location x and y
                    location_x = (location / state.size) - last_move_x
                    location_y = (location % state.size) - last_move_y
                
                    # calculate 12d response pattern hash
                    hash_pattern  = hash_base + location_x
                    hash_pattern *= _HASHVALUE
                    hash_pattern += location_y

                    # dictionary lookup
                    id = self.pattern_response_12d.get( hash_pattern )

                    if id >= 0:

                        tensor[ offSet + id, location ] = 1

        return offSet + self.pattern_response_12d_size


    @cython.nonecheck(False)
    cdef int get_non_response_3x3(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet):
        """
           Set 3x3 hash pattern for every legal location where
           pattern lookup value is being set instead of 1
        """

        # TODO tensor type has to be float

        return offSet + 1


    @cython.nonecheck(False)
    cdef int get_non_response_3x3_offset(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet):
        """
           Set 3x3 hash pattern for every legal location where
           #pattern_id is offset
        """

        cdef short i, location
        cdef int   id

        # loop over all legal moves and set to one
        for i in range(state.moves_legal.count):

            # get location
            location = state.moves_legal.locations[ i ]
            # get location hash and dict lookup
            id = self.pattern_non_response_3x3.get( state.get_3x3_hash( location ) )

            if id >= 0:

                tensor[ offSet + id, location ] = 1

        return offSet + self.pattern_non_response_3x3_size


    @cython.nonecheck(False)
    cdef int zeros(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet):
        """
           Plane filled with zeros
        """

        ##########################################################
        # strange things happen if a function does not do anything
        # do not remove next line without extensive testing!!!!!!!
        tensor[ offSet, 0 ] = 0

        return offSet + 1


    @cython.nonecheck(False)
    cdef int ones(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet):
        """
           Plane filled with ones
        """

        cdef short location

        for location in range(0, self.board_size):

            tensor[ offSet, location ] = 1
        return offSet + 1


    @cython.nonecheck(False)
    cdef int colour(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet):
        """
           Value net feature, plane with ones if active_player is black else zeros
        """

        cdef short location

        # if player_current is white
        if state.player_current == _BLACK:

                for location in range(0, self.board_size):

                    tensor[ offSet, location ] = 1

        return offSet + 1


    @cython.nonecheck(False)
    cdef int ko(self, GameState state, tensor_type[ :, ::1 ] tensor, int offSet):
        """
           Single plane encoding ko location
        """

        if state.ko is not _PASS:

            tensor[ offSet, state.ko ] = 1

        return offSet + 1


    ############################################################################
    #   init function                                                          #
    #                                                                          #
    ############################################################################


    def __init__(self, list feature_list, char size=19, dict_nakade=None, dict_3x3=None, dict_12d=None, verbose=False):
        """
        """

        self.size       = size
        self.board_size = size * size

        cdef int i

        # preprocess_method is a function pointer:
        # ctypedef int (*preprocess_method)(Preprocess, GameState, tensor_type[ :, ::1 ], char*, int)
        cdef preprocess_method processor

        # create a list with function pointers
        self.processors = <preprocess_method  *>malloc(len(feature_list) * sizeof(preprocess_method))

        if not self.processors:
            raise MemoryError()

        # load nakade patterns
        self.pattern_nakade = {}
        self.pattern_nakade_size = 0
        if dict_nakade is not None:
            with open(dict_nakade, 'r') as f:
                s = f.read()
                self.pattern_nakade = ast.literal_eval(s)
                self.pattern_nakade_size = max(self.pattern_nakade.values()) + 1

        # load 12d response patterns
        self.pattern_response_12d = {}
        self.pattern_response_12d_size = 0
        if dict_12d is not None:
            with open(dict_12d, 'r') as f:
                s = f.read()
                self.pattern_response_12d = ast.literal_eval(s)
                self.pattern_response_12d_size = max(self.pattern_response_12d.values()) + 1

        # load 3x3 non response patterns
        self.pattern_non_response_3x3 = {}
        self.pattern_non_response_3x3_size = 0
        if dict_3x3 is not None:
            with open(dict_3x3, 'r') as f:
                s = f.read()
                self.pattern_non_response_3x3 = ast.literal_eval(s)
                self.pattern_non_response_3x3_size = max(self.pattern_non_response_3x3.values()) + 1

        if verbose:
            print("loaded " + str(self.pattern_nakade_size) + " nakade patterns")
            print("loaded " + str(self.pattern_response_12d_size) + " 12d patterns")
            print("loaded " + str(self.pattern_non_response_3x3_size) + " 3x3 patterns")

        self.feature_list = feature_list
        self.output_dim = 0

        # loop over feature_list add the corresponding function
        # and increment output_dim accordingly
        for i in range(len(feature_list)):
            feat = feature_list[ i ].lower()
            if feat == "board":
                processor            = self.get_board
                self.output_dim     += 3

            elif feat == "ones":
                processor            = self.ones
                self.output_dim     += 1

            elif feat == "turns_since":
                processor            = self.get_turns_since
                self.output_dim     += 8

            elif feat == "liberties":
                processor            = self.get_liberties
                self.output_dim     += 8

            elif feat == "ladder_capture":
                processor            = self.get_ladder_capture
                self.output_dim     += 1

            elif feat == "ladder_escape":
                processor            = self.get_ladder_escape
                self.output_dim     += 1

            elif feat == "sensibleness":
                processor            = self.get_sensibleness
                self.output_dim     += 1

            elif feat == "zeros":
                processor            = self.zeros
                self.output_dim     += 1

            elif feat == "legal":
                processor            = self.get_legal
                self.output_dim     += 1

            elif feat == "response":
                processor            = self.get_response
                self.output_dim     += 1

            elif feat == "save_atari":
                processor            = self.get_save_atari
                self.output_dim     += 1

            elif feat == "neighbor":
                processor            = self.get_neighbor
                self.output_dim     += 2

            elif feat == "nakade":
                processor            = self.get_nakade
                self.output_dim     += self.pattern_nakade_size

            elif feat == "response_12d":
                processor            = self.get_response_12d
                self.output_dim     += self.pattern_response_12d_size

            elif feat == "non_response_3x3":
                processor            = self.get_non_response_3x3
                self.output_dim     += self.pattern_non_response_3x3_size

            elif feat == "color":
                processor            = self.colour
                self.output_dim     += 1

            elif feat == "ko":
                processor            = self.ko
                self.output_dim     += 1
            else:

                # incorrect feature input
                raise ValueError("uknown feature: %s" % feat)

            self.processors[ i ] = processor


    def __dealloc__(self):
        """
           Prevent memory leaks by freeing all arrays created with malloc
        """

        if self.processors is not NULL:
            free(self.processors)

    ############################################################################
    #   public cdef function                                                   #
    #                                                                          #
    ############################################################################


    @cython.nonecheck(False)
    cdef np.ndarray[ tensor_type, ndim=4 ] generate_tensor(self, GameState state):
        """
           Convert a GameState to a Theano-compatible tensor
        """

        cdef int i
        cdef preprocess_method proc

        # create complete array now instead of concatenate later
        # TODO check if we can use a Malloc array somehow.. faster!!
        cdef np.ndarray[ tensor_type, ndim=2 ] np_tensor = np.zeros((self.output_dim, self.board_size), dtype=np.int8)
        cdef tensor_type[ :, ::1 ] tensor                = np_tensor

        cdef int offSet = 0

        # loop over all processors and generate tensor
        for i in range(len(self.feature_list)):

            proc   = self.processors[ i ]
            offSet = proc(self, state, tensor, offSet)

        # create a singleton 'batch' dimension
        return np_tensor.reshape((1, self.output_dim, self.size, self.size))


    ############################################################################
    #   public def function (Python)                                           #
    #                                                                          #
    ############################################################################


    def state_to_tensor(self, GameState state):
        """
           Convert a GameState to a Theano-compatible tensor
        """

        return self.generate_tensor(state)


    def get_output_dimension(self):
        """
           return output_dim, the amount of planes an output tensor will have
        """

        return self.output_dim


    def get_feature_list(self):
        """
           return feature list
        """

        return self.feature_list
