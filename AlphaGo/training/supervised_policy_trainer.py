import os
import json
import threading
import h5py as h5
import numpy as np
from keras import backend as K
from AlphaGo.util import confirm
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, Callback
from AlphaGo.models.policy import CNNPolicy
from AlphaGo.preprocessing.preprocessing import Preprocess

# default settings
DEFAULT_MAX_VALIDATION = 1e9
DEFAULT_TRAIN_VAL_TEST = [.95, .05, .0]
DEFAULT_LEARNING_RATE = 3e-3
DEFAULT_BATCH_SIZE = 16
DEFAULT_DECAY = 1.25e-8
DEFAULT_EPOCH = 10

FILE_METADATA = 'metadata.json'
FOLDER_WEIGHT = 'weights'

# shuffle files
FILE_VALIDATE = 'shuffle_validate.npz'
FILE_TRAIN = 'shuffle_train.npz'
FILE_TEST = 'shuffle_test.npz'

# Map from transform name to function.
BOARD_TRANSFORMATIONS = {
    "noop": lambda feature: feature,
    "rot90": lambda feature: np.rot90(feature, 1),
    "rot180": lambda feature: np.rot90(feature, 2),
    "rot270": lambda feature: np.rot90(feature, 3),
    "fliplr": lambda feature: np.fliplr(feature),
    "flipud": lambda feature: np.flipud(feature),
    "diag1": lambda feature: np.transpose(feature),
    "diag2": lambda feature: np.fliplr(np.rot90(feature, 1))
}
# Ordered list of transform functions.
ORDERED_TRANSFORMATIONS = [BOARD_TRANSFORMATIONS[key] for key in
                           sorted(BOARD_TRANSFORMATIONS.keys())]
# Map from transform name to index.
TRANSFORMATION_INDICES = {name: i for (i, name) in enumerate(sorted(BOARD_TRANSFORMATIONS.keys()))}


def one_hot_action(action, size=19):
    """Helper function to convert an (x, y) action into a size x size array of zeros with a 1 at
       (x, y).
    """
    categorical = np.zeros((size, size))
    categorical[action] = 1
    return categorical


class ShuffledHDF5BatchGenerator:
    """A generator of batches of training data for use with the fit_generator function of Keras.
       Data are accessed from the given set of indices.

       Each row of `indices` must have an index into the dataset in the 0th column and a
       transform index in the 1st column.

       If a `reshuffle_seed` is given, the indices are pseudo-randomly reshuffled aftear each pass
       through the full dataset. If `initial_batch` is given, then the generator picks up 'where it
       left off'. If `initial_batch` is greater than the number of batches in the dataset, the
       indices will be reshuffled `int(initial_batch / batches_per_data_epoch)` times.

       It is threading safe but not multiprocessing. Only use it with `pickle_safe=False` when using
       multiple workers.
    """

    def __init__(self, state_dataset, action_dataset, indices, batch_size, initial_batch=0,
                 reshuffle_seed=None):
        self.state_dataset = state_dataset
        self.action_dataset = action_dataset
        self.n_data = len(state_dataset)
        # lock used for multithreaded workers
        self.lock = threading.Lock()
        self.indices_max = len(indices)
        self.batch_size = batch_size
        self.indices = indices
        self.batch = initial_batch
        self.seed = reshuffle_seed
        self.batches_per_data_epoch = int(self.n_data / self.batch)

        if self.seed is not None:
            # Shuffle the data once for each pass through the data that has 'already elapsed'
            for _ in range(int(initial_batch / self.batches_per_data_epoch)):
                self.reshuffle()

    def __iter__(self):
        return self

    def reshuffle(self):
        """Reshuffle self.indices pseudo-randomly, using self.seed to make it deterministic.

        Note that the seed is set in each call, so the same shuffle is used each time; this does not
        immediately "undo" itself, but creates a large cycle that will eventually be repeated.
        """
        np.random.seed(self.seed)
        # Note that shuffling only permutes rows, keeping each (index, transform) pair intact.
        np.random.shuffle(self.indices)

    def next(self):
        """Iterator function, returning a minibatch of (state, action) pairs.
        """
        # Lock must be used for hdf5 access and for computing batch indices.
        with self.lock:
            # get indices into data of next minibatch using 'wrapped' indexing.
            batch_indices = range(self.batch * self.batch_size, (self.batch + 1) * self.batch_size)
            sample_indices = np.take(self.indices, batch_indices, axis=0, mode='wrap')
            self.batch += 1

            # Re-shuffle the order of the dataset after each full pass through it.
            if self.seed is not None and (self.batch % self.batches_per_data_epoch) == 0:
                self.reshuffle()

            states = self.state_dataset[sample_indices[:, 0], :]
            actions_xy = self.action_dataset[sample_indices[:, 0], :]

        game_size = states.shape[-1]
        actions_onehot = np.zeros((self.batch_size, game_size * game_size))

        for i in range(self.batch_size):
            # Get rotation/reflection symmetry of this sample.
            transform = ORDERED_TRANSFORMATIONS[sample_indices[i][1]]

            # Transform each plane of the sample. Note that using loop comprehension causes the
            # transformation to act on the 3rd and 4th dimensions (since they are the non-singleton
            # dimensions), which are the board rows and columns.
            states[i] = np.array([transform(plane) for plane in states[i]])
            actions_onehot[i] = transform(one_hot_action(actions_xy[i], game_size)).flatten()

        return (states, actions_onehot)


class MetadataWriterCallback(Callback):

    def __init__(self, path, initial_data={}):
        super(MetadataWriterCallback, self).__init__()
        self.file = path
        self.metadata = {
            "epoch_logs": [],
            "best_epoch": 0,
            "current_epoch": 0,
            "current_batch": 0
        }
        self.metadata.update(initial_data)

    def on_epoch_end(self, epoch, logs={}):
        # Save current epoch number. When restarting, 'epoch' counts from zero and is therefore
        # unreliable. The true number of epochs that have passed is in len(epoch_logs)
        self.metadata["current_epoch"] = len(self.metadata["epoch_logs"])
        # Append logs to metadata.
        self.metadata["epoch_logs"].append(logs)

        if "val_loss" in logs:
            key = "val_loss"
        else:
            key = "loss"

        best_loss = self.metadata["epoch_logs"][self.metadata["best_epoch"]][key]
        if logs.get(key) < best_loss:
            self.metadata["best_epoch"] = epoch

        # save meta to file
        with open(self.file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def on_batch_begin(self, batch, logs={}):
        # Batches are counted here, but only saved in `on_epoch_end`. This is deliberate; when
        # resuming training, 'current_batch' should reflect the batch number of the most recently
        # saved weights file, which is itself only saved at the end of an epoch.
        self.metadata["current_batch"] += 1


class LrDecayCallback(Callback):
    """Set learning rate every batch according to:
       initial_learning_rate * (1. / (1. + self.decay * curent_batch))
    """

    def __init__(self, learning_rate, decay, initial_batch=0):
        super(LrDecayCallback, self).__init__()
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_batch = initial_batch

    def set_lr(self):
        # calculate learning rate
        new_lr = self.learning_rate * (1. / (1. + self.decay * self.current_batch))

        # set new learning rate
        K.set_value(self.model.optimizer.lr, new_lr)

    def on_batch_begin(self, batch, logs={}):
        # Note: using `batch` as passed in is not useful as it starts at 0 every epoch.

        # Set new learning rate.
        self.set_lr()

        # Increment current_batch counter.
        self.current_batch += 1


class LrStepDecayCallback(Callback):
    """Set learning rate every decay_every batches according to:
       initial_learning_rate * (decay ^ (current_batch / decay_every))
    """

    def __init__(self, learning_rate, decay, decay_every, initial_batch=0):
        super(LrStepDecayCallback, self).__init__()
        self.learning_rate = learning_rate
        self.decay_every = decay_every
        self.decay = decay
        self.current_batch = initial_batch

    def set_lr(self):
        # Calculate learning rate.
        n_decay = int(self.current_batch / self.decay_every)
        new_lr = self.learning_rate * (self.decay ** n_decay)

        # Set new learning rate.
        K.set_value(self.model.optimizer.lr, new_lr)

    def on_train_begin(self, logs={}):
        # Set initial learning rate (necessary here in case initial_batch > 0 in the constructor,
        # since `set_lr` is not called in every `on_batch_begin`).
        self.set_lr()

    def on_batch_begin(self, batch, logs={}):
        # Note: using `batch` as passed in is not useful as it starts at 0 every epoch.

        # Change learning rate in discrete steps.
        if self.self.current_batch % self.decay_every == 0:
            self.set_lr()

        # Increment current_batch counter.
        self.current_batch += 1


class ModelCheckpointWithOffset(ModelCheckpoint):
    """A ModelCheckpoint that adds an offset to the epoch before saving.
    """
    def __init__(self, offset, *args, **kwargs):
        super(ModelCheckpointWithOffset, self).__init__(*args, **kwargs)
        self.offset = offset

    def on_epoch_end(self, epoch, logs={}):
        super(ModelCheckpointWithOffset, self).on_epoch_end(self.offset + epoch, logs)


def validate_feature_planes(verbose, dataset, model_features):
    """Verify that dataset's features match the model's expected features.
    """

    if 'features' in dataset:
        dataset_features = dataset['features'][()]
        dataset_features = dataset_features.split(",")
        if len(dataset_features) != len(model_features) or \
           any(df != mf for (df, mf) in zip(dataset_features, model_features)):
            raise ValueError("Model JSON file expects features \n\t%s\n"
                             "But dataset contains \n\t%s" % ("\n\t".join(model_features),
                                                              "\n\t".join(dataset_features)))
        elif verbose:
            print("Verified that dataset features and model features exactly match.")
    else:
        # Cannot check each feature, but can check number of planes.
        n_dataset_planes = dataset["states"].shape[1]
        tmp_preprocess = Preprocess(model_features)
        n_model_planes = tmp_preprocess.output_dim
        if n_dataset_planes != n_model_planes:
            raise ValueError("Model JSON file expects a total of %d planes from features \n\t%s\n"
                             "But dataset contains %d planes" % (n_model_planes,
                                                                 "\n\t".join(model_features),
                                                                 n_dataset_planes))
        elif verbose:
            print("Verified agreement of number of model and dataset feature planes, but cannot "
                  "verify exact match using old dataset format.")


def load_indices_from_file(shuffle_file):
    # load indices from shuffle_file
    with open(shuffle_file, "r") as f:
        indices = np.load(f)
    return indices


def save_indices_to_file(shuffle_file, indices):
    # save indices to shuffle_file
    with open(shuffle_file, "w") as f:
        np.save(f, indices)


def create_and_save_shuffle_indices(train_val_test, max_validation, n_total_data_size,
                                    shuffle_file_train, shuffle_file_val, shuffle_file_test):
    """Create an array with all unique state and symmetry pairs, then split it into test/validation/
       training sets, and save each to a seperate file.
    """

    symmetries = TRANSFORMATION_INDICES.values()
    n_symmetries = len(symmetries)
    n_data_with_symmetries = n_total_data_size * n_symmetries

    # Create an array with a unique row for each combination of a training example and a symmetry.
    # shuffle_indices[i][0] is an index into training examples,
    # shuffle_indices[i][1] is the index (from 0 to 7) of the symmetry transformation to apply
    shuffle_indices = np.empty(shape=(n_data_with_symmetries, 2), dtype=int)
    for dataset_idx in range(n_total_data_size):
        for symmetry_idx in symmetries:
            shuffle_indices[dataset_idx * n_symmetries + symmetry_idx][0] = dataset_idx
            shuffle_indices[dataset_idx * n_symmetries + symmetry_idx][1] = symmetries[symmetry_idx]

    # Shuffle rows without affecting x,y pairs.
    np.random.shuffle(shuffle_indices)

    # Validation set size.
    n_val_data = int(train_val_test[1] * n_data_with_symmetries)
    # Limit validation set to max_validation.
    if n_val_data > max_validation:
        n_val_data = max_validation

    # Test set size.
    n_test_data = int(train_val_test[2] * n_data_with_symmetries)

    # Train set size.
    n_train_data = n_data_with_symmetries - n_val_data - n_test_data

    # Create training set and save to file shuffle_file_train.
    train_indices = shuffle_indices[0:n_train_data]
    save_indices_to_file(shuffle_file_train, train_indices)

    # create validation set and save to file shuffle_file_val
    val_indices = shuffle_indices[n_train_data:n_train_data + n_val_data]
    save_indices_to_file(shuffle_file_val, val_indices)

    # create test set and save to file shuffle_file_test
    test_indices = shuffle_indices[n_train_data + n_val_data:
                                   n_train_data + n_val_data + n_test_data]
    save_indices_to_file(shuffle_file_test, test_indices)


def load_train_val_test_indices(arg_symmetries, batch_size, directory, verbose, dataset_length):
    """Load shuffling indices from .npz files and processes them for the current coniguration.
       Namely, remove unwanted symmerties (as specified by arg_symmetries) and makes train set size
       divisible by batch_size.

       Returns tuple of indices for train/val/test sets.
    """
    # shuffle file locations for train/validation/test set
    shuffle_file_train = os.path.join(directory, FILE_TRAIN)
    shuffle_file_test = os.path.join(directory, FILE_TEST)
    shuffle_file_val = os.path.join(directory, FILE_VALIDATE)

    # load from .npz files
    train_indices = load_indices_from_file(shuffle_file_train)
    test_indices = load_indices_from_file(shuffle_file_test)
    val_indices = load_indices_from_file(shuffle_file_val)

    if dataset_length != len(train_indices) + len(val_indices) + len(test_indices):
        raise ValueError("Mismatch in dataset size and size of saved indices.")

    # used symmetries
    if arg_symmetries == "all":
        # add all symmetries
        symmetries = TRANSFORMATION_INDICES.values()
    elif arg_symmetries == "none":
        # only add standart orientation
        symmetries = [TRANSFORMATION_INDICES["noop"]]
    else:
        # add specified symmetries
        symmetries = [TRANSFORMATION_INDICES[name] for name in arg_symmetries.strip().split(",")]

    def remove_unused_symmetries(indices, symmetries):
        return indices[np.in1d(indices[:, 1], symmetries), :]

    # Remove from each set symmetries that are not used during current run.
    if len(symmetries) != len(TRANSFORMATION_INDICES):
        train_indices = remove_unused_symmetries(train_indices, symmetries)
        test_indices = remove_unused_symmetries(test_indices, symmetries)
        val_indices = remove_unused_symmetries(val_indices, symmetries)

    # Need to make sure training set size is divisible by minibatch size for the batch generator.
    n_remove = len(train_indices) % batch_size
    train_indices = train_indices[n_remove:]

    if verbose:
        print("Dataset and indices loaded.")
        print("\t%d total positions" % dataset_length)
        print("\t%d total samples (with symmetries)" % (dataset_length * len(symmetries)))
        print("\t%d training samples (cut %d to make multiple of batch size)" %
              (len(train_indices), n_remove))
        print("\t%d validation samples" % len(val_indices))
        print("\t%d test samples" % len(test_indices))

    return train_indices, val_indices, test_indices


def validate_args(args):
    """Save all args to metadata, check if critical settings have been changed and prompt user about
       it. Creates new shuffle files if needed.

       Returns (metdata, resume_weights)
    """
    # Resuming with defaults if args.mode is 'resume', or with sanity-checks on all settings if
    # in 'train' mode but 'weights' are set.
    resume = args.mode == 'resume' or args.weights is not None

    # Shuffle file locations for train/validation/test set.
    shuffle_file_train = os.path.join(args.out_directory, FILE_TRAIN)
    shuffle_file_test = os.path.join(args.out_directory, FILE_TEST)
    shuffle_file_val = os.path.join(args.out_directory, FILE_VALIDATE)

    # Determine if new shuffle files have to be created.
    save_new_shuffle_indices = not resume

    if resume:
        if args.verbose:
            print("Resuming from %s with weights %s" % (args.out_directory, args.weights))

        # Load previous metadata.
        meta_file = os.path.join(args.out_directory, FILE_METADATA)
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                metadata = json.load(f)
            if args.verbose:
                print("Previous metadata loaded: %d epochs. New epochs will be appended." %
                      len(metadata["epoch_logs"]))
        else:
            raise ValueError("Cannot resume training without metadata.json file.")

        # Sanity-check consistency if in 'train' mode.
        if args.mode == 'train':
            def maybe_confirm(prompt, default):
                return args.override or confirm(prompt, default)

            # Check if model is the same.
            if metadata["model_file"] != args.model:
                print("The model file is different from the model file used last run: " +
                      metadata["model_file"] + ". It might be different than the old one.")
                if not maybe_confirm("Are you sure you want to use the new model?", False):  # noqa: E501
                    raise ValueError("User abort after mismatch model files.")
                metadata["model_file"] = args.model

            # Check if decay_every is the same.
            if metadata["decay_every"] != args.decay_every:
                print("Setting a new --decay-every might result in a different learning rate, restarting training might be advisable.")  # noqa: E501
                print("Previously, --decay-every was set to %d" % metadata["decay_every"])
                if not maybe_confirm("Are you sure you want to use new decay every setting?", False):  # noqa: E501
                    raise ValueError("User abort after mismatch in decay_every")
                metadata["decay_every"] = args.decay_every

            # Check if learning_rate is the same.
            if metadata["learning_rate"] != args.learning_rate:
                print("Setting a new --learning-rate might result in a different learning rate, restarting training might be advisable.")  # noqa: E501
                print("Previously, --learning-rate was set to %d" % metadata["learning_rate"])
                if not maybe_confirm("Are you sure you want to use new learning rate setting?", False):  # noqa: E501
                    raise ValueError("User abort after mismatch in learning_rate")
                metadata["learning_rate"] = args.learning_rate

            # Check if decay is the same.
            if metadata["decay"] != args.decay:
                print("Setting a new --decay might result in a different learning rate, restarting training might be advisable.")  # noqa: E501
                print("Previously, --decay was set to %d" % metadata["decay"])
                if not maybe_confirm("Are you sure you want to use new decay setting?", False):  # noqa: E501
                    raise ValueError("User abort after mismatch in decay")
                metadata["decay"] = args.decay

            # Check if batch_size is the same.
            if metadata["batch_size"] != args.minibatch:
                print("current_batch will be recalculated, restarting training might be advisable.")
                print("Previously, --minibatch was set to %d" % metadata["batch_size"])
                if not maybe_confirm("Are you sure you want to use new minibatch setting?", False):
                    raise ValueError("User abort after mismatch in batch_size")
                total_samples_seen = metadata["current_batch"] * metadata["batch_size"]
                metadata["current_batch"] = int(total_samples_seen / args.minibatch)
                metadata["batch_size"] = args.minibatch

            # Check if max_validation is the same.
            if metadata["max_validation"] != args.max_validation:
                print("Training set and validation set should be strictly separated, setting a new fraction will generate a new validation set that might include data from the current traing set.")  # noqa: E501
                print("We recommend you use the previous max-validation setting: %d" % metadata["max_validation"])  # noqa: E501
                if not maybe_confirm("Are you sure you want to use new max-validation setting?", False):  # noqa: E501
                    raise ValueError("User abort after mismatch in max_validation")
                metadata["max_validation"] = args.max_validation
                # New shuffle files have to be created.
                save_new_shuffle_indices = True

            # Check if train_val_test is the same.
            if metadata["train_val_test"] != args.train_val_test:
                print("Training set and validation set should be strictly separated, setting a new fraction will generate a new validation set that might include data from the current traing set.")  # noqa: E501
                print("We recommend you use the previous values: %f %f %f" % metadata["train_val_test"])  # noqa: E501
                if not maybe_confirm("Are you sure you want to use new train-val-test fraction setting?", False):  # noqa: E501
                    raise ValueError("User abort after mismatch in train_val_test")
                metadata["train_val_test"] = args.train_val_test
                # New shuffle files have to be created.
                save_new_shuffle_indices = True

            # Check that the same datset is being used.
            if metadata["training_data"] != args.train_data:
                print("Changing the dataset will overwrite previous shuffle indices.")
                print("Previously, the training set was " + metadata["training_data"])
                if not maybe_confirm("Are you sure you want to resume training with a different dataset?", False):  # noqa: E501
                    raise ValueError("User abort after mismatch in training data file")
                metadata["training_data"] = args.train_data
                save_new_shuffle_indices = True

            # Check if symmetries are the same - no errors if changed.
            if metadata["symmetries"] != args.symmetries:
                print("Changing symmetries from %s to %s" % (metadata["symmetries"],
                                                             args.symmetries))
                metadata["symmetries"] = args.symmetries

            # Check if reshuffle_seed is the same - no errors if changed.
            if metadata["reshuffle_seed"] != args.reshuffle_seed:
                print("Changing reshuffling seed from %d to %d" % (metadata["reshuffle_seed"],
                                                                   args.reshuffle_seed))
                metadata["reshuffle_seed"] = args.reshuffle_seed

            # Check if epoch_length is the same - no errors if changed.
            if metadata["epoch_length"] != args.epoch_length:
                print("Changing epoch length from %d to %d" % (metadata["epoch_length"],
                                                               args.epoch_length))
                metadata["epoch_length"] = args.epoch_length

        # Check if epochs is the same - no errors if changed. (Note that args.epochs may be set even
        # when mode is 'resume'). If 'None', simply uses previous value.
        if args.epochs is not None and metadata["epochs"] != args.epochs:
            print("Changing total epochs from %d to %d" % (metadata["epochs"], args.epochs))
            metadata["epochs"] = args.epochs

        # Resume weights defaults to the most recently-saved weights.
        resume_weights = args.weights or \
            "weights.{epoch:05d}.hdf5".format(epoch=metadata["current_epoch"])

        # Check if shuffle files exist.
        if not all(map(os.path.exists, [shuffle_file_train, shuffle_file_val, shuffle_file_test])):
            if not maybe_confirm("WARNING! Shuffle files not found. Recreate and continue?", False):
                raise ValueError("User abort for missing shuffle files.")
            save_new_shuffle_indices = True
    else:
        # Starting from scratch - create new metadata.
        if args.verbose:
            print("Starting with new metadata.")

        resume_weights = None

        if os.path.exists(args.out_directory):
            if args.verbose:
                if not confirm("Ouptut directory exists. Overwrite previous results?", False):
                    raise ValueError("User abort - not overwriting output directory.")
                else:
                    print("Starting fresh output directory %s" % args.out_directory)
            os.makedirs(args.out_directory)

        metadata = {
            "model_file": args.model,
            "decay_every": args.decay_every,
            "training_data": args.train_data,
            "learning_rate": args.learning_rate,
            "decay": args.decay,
            "batch_size": args.batch_size,
            "max_validation": args.max_validation,
            "reshuffle_seed": args.reshuffle_seed,
            "train_val_test": args.train_val_test,
            "epochs": args.epochs,
            "epoch_length": args.epoch_length,
            "symmetries": args.symmetries
        }

    # Record all command line args in a list so that all args are recorded even when training is
    # stopped and resumed.
    meta_args = metadata.get("cmd_line_args", [])
    meta_args.append(vars(args))
    metadata["cmd_line_args"] = meta_args

    # Open dataset file just to get length.
    with h5.File(metadata["training_data"]) as dataset:
        dataset_length = len(dataset)

    # Create and save new shuffle indices if needed.
    if save_new_shuffle_indices:
        create_and_save_shuffle_indices(metadata["train_val_test"],
                                        metadata["max_validation"],
                                        dataset_length,
                                        shuffle_file_train,
                                        shuffle_file_val,
                                        shuffle_file_test,
                                        args.verbose)

        if args.verbose:
            print("created new data shuffling indices")

    return metadata, resume_weights


def train(verbose, out_directory, metadata, initial_weights=None):
    # Load model from json spec.
    policy = CNNPolicy.load_model(metadata["model_file"])
    model_features = policy.preprocessor.feature_list
    model = policy.model

    # Load initial weights.
    if initial_weights is not None:
        model.load_weights(os.path.join(out_directory, FOLDER_WEIGHT, initial_weights))

    # Load dataset.
    dataset = h5.File(metadata["training_data"])

    # Verify that dataset's features match the model's expected features.
    validate_feature_planes(verbose, dataset, model_features)

    # Create callbacks.
    meta_writer = MetadataWriterCallback(os.path.join(out_directory, FILE_METADATA), metadata)
    # Update 'metadata' with fields created by the callback.
    metadata.update(meta_writer.metadata)
    model_saver = ModelCheckpointWithOffset(offset=metadata["current_epoch"])

    # Get train/validation/test indices.
    train_indices, val_indices, test_indices = load_train_val_test_indices(metadata["symmetries"],
                                                                           metadata["batch_size"],
                                                                           out_directory,
                                                                           verbose,
                                                                           dataset_length)

    # Create dataset generators. The training generator will be reshuffled after each pass through
    # the full dataset but the validation generator will not.
    train_data_generator = ShuffledHDF5BatchGenerator(
        dataset["states"],
        dataset["actions"],
        train_indices,
        metadata["batch_size"],
        initial_batch=metadata["current_batch"],
        reshuffle_seed=metadata["reshuffle_seed"])
    val_data_generator = ShuffledHDF5BatchGenerator(
        dataset["states"],
        dataset["actions"],
        val_indices,
        metadata["batch_size"])

    # Choose learning rate decay callback.
    if metadata["decay_every"] is None:
        # Use normal decay without momentum.
        lr_scheduler_callback = LrDecayCallback(metadata["learning_rate"],
                                                metadata["decay"],
                                                metadata["current_batch"])
    else:
        # Use 'step' decay.
        lr_scheduler_callback = LrStepDecayCallback(metadata["learning_rate"],
                                                    metadata["decay"],
                                                    metadata["decay_every"],
                                                    metadata["current_batch"])

    sgd = SGD(lr=metadata["learning_rate"])
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

    remaining_epochs = metadata["epochs"] - metadata["current_epoch"]
    if remaining_epochs <= 0:
        raise ValueError("No more epochs to train!")

    if verbose:
        print("STARTING TRAINING")

    model.fit_generator(
        generator=train_data_generator,
        samples_per_epoch=metadata["epoch_length"],
        nb_epoch=remaining_epochs,
        callbacks=[meta_writer, model_saver, lr_scheduler_callback],
        validation_data=val_data_generator,
        nb_val_samples=len(val_indices))


def handle_arguments(cmd_line_args=None):
    """Run training. command-line args may be passed in as a list of strings.
    """

    import argparse
    parser = argparse.ArgumentParser(description='Perform supervised training on a policy network.')

    # Subparser is always first argument.
    subparsers = parser.add_subparsers(help='sub-command help', dest='mode')

    # Add 'global' arguments, i.e. those used for both 'train' and 'resume'
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501

    # Sub parser for starting training.
    train = subparsers.add_parser('train', help='Start or resume supervised training on a policy network.')  # noqa: E501
    # Required arguments.
    train.add_argument("out_directory", help="directory where metadata and weights will be saved")  # noqa: E501
    train.add_argument("model", help="Path to a JSON model file (i.e. from CNNPolicy.save_model())")  # noqa: E501
    train.add_argument("train_data", help="A .h5 file of training data")
    # Frequently used args.
    train.add_argument("--minibatch", "-B", help="Size of training data minibatches. Default: " + str(DEFAULT_BATCH_SIZE), type=int, default=DEFAULT_BATCH_SIZE)  # noqa: E501
    train.add_argument("--epochs", "-E", help="Total number of iterations on the data. Default: " + str(DEFAULT_EPOCH), type=int, default=DEFAULT_EPOCH)  # noqa: E501
    train.add_argument("--epoch-length", "-l", help="Number of training examples considered 'one epoch'. Default: # training data", type=int, default=None)  # noqa: E501
    train.add_argument("--learning-rate", "-r", help="Learning rate - how quickly the model learns at first. Default: " + str(DEFAULT_LEARNING_RATE), type=float, default=DEFAULT_LEARNING_RATE)  # noqa: E501
    train.add_argument("--decay", "-d", help="The rate at which learning decreases. Default: " + str(DEFAULT_DECAY), type=float, default=DEFAULT_DECAY)  # noqa: E501
    train.add_argument("--decay-every", "-de", help="Use step-decay: decay --learning-rate with --decay every --decay-every batches. Default: None", type=int, default=None)  # noqa: E501
    train.add_argument("--override", help="Turn on prompt override mode", default=False, action="store_true")  # noqa: E501
    # Slightly fancier args.
    train.add_argument("--weights", help="Name of a .h5 weights file (in the output directory) to load to resume training", default=None)  # noqa: E501
    train.add_argument("--train-val-test", help="Fraction of data to use for training/val/test. Must sum to 1. Default: " + str(DEFAULT_TRAIN_VAL_TEST), nargs=3, type=float, default=DEFAULT_TRAIN_VAL_TEST)  # noqa: E501
    train.add_argument("--max-validation", help="maximum validation set size. default: " + str(DEFAULT_MAX_VALIDATION), type=int, default=DEFAULT_MAX_VALIDATION)  # noqa: E501
    train.add_argument("--symmetries", help="none, all or comma-separated list of transforms, subset of: noop,rot90,rot180,rot270,fliplr,flipud,diag1,diag2. Default: all", default='all')  # noqa: E501
    train.add_argument("--reshuffle-seed", help="Integer seed for reshuffling after each pass through the training set.", default=8675309)  # noqa: E501

    # Sub parser for resuming training.
    resume = subparsers.add_parser('resume', help='Resume supervised training on a policy network. (Settings are loaded from savefile.)')  # noqa: E501
    # Required arguments.
    resume.add_argument("out_directory", help="directory where metadata and weight files where stored during previous session.")  # noqa: E501
    # Optional arguments (if not specified, previous values will be used).
    resume.add_argument("--weights", help="Name of a .h5 weights file (in the output directory) to load to resume training. Default: #Newest weight file.", default=None)  # noqa: E501
    resume.add_argument("--epochs", "-E", help="Total number of iterations on the data. Default: #Epochs set on previous run", type=int, default=None)  # noqa: E501

    # Show help or parse arguments.
    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    # Validate arguments.
    metadata, resume_weights = validate_args(args)

    # Run training. Note that resume_weights may be None.
    train(args.verbose, args.out_directory, metadata, initial_weights=resume_weights)


if __name__ == '__main__':
    handle_arguments()
