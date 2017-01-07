import os
from AlphaGo.training.supervised_policy_trainer import handle_arguments
from AlphaGo.training.supervised_policy_trainer import load_indices_from_file
from AlphaGo.training.supervised_policy_trainer import create_and_save_shuffle_indices
import unittest
import numpy as np
import numpy.testing as npt


class TestSupervisedPolicyTrainer(unittest.TestCase):
    def testTrain(self):
        model = 'tests/test_data/minimodel.json'
        data = 'tests/test_data/hdf5/alphago-vs-lee-sedol-features.hdf5'
        output = 'tests/test_data/.tmp.training/'
        args = ['train', output, model, data, '--epochs', '1', '--epoch-length', '160']
        handle_arguments(args)

        # remove temporary files
        os.remove(os.path.join(output, 'metadata.json'))
        os.remove(os.path.join(output, 'shuffle_policy_train.npz'))
        os.remove(os.path.join(output, 'shuffle_policy_validate.npz'))
        os.remove(os.path.join(output, 'shuffle_policy_test.npz'))
        os.remove(os.path.join(output, 'weights', 'weights.00000.hdf5'))
        os.rmdir(os.path.join(output, 'weights'))
        os.rmdir(output)

    def testResumeLearning(self):
        model = 'tests/test_data/minimodel.json'
        data = 'tests/test_data/hdf5/alphago-vs-lee-sedol-features.hdf5'
        output = 'tests/test_data/.tmp.resume-training/'
        args = ['train', output, model, data, '--epochs', '1', '--epoch-length', '160']
        handle_arguments(args)

        # resume learning
        args = ['train', output, model, data, '--epochs', '2', '--epoch-length', '160',
                '--weights', 'weights.00000.hdf5']
        handle_arguments(args)

        # remove temporary files
        os.remove(os.path.join(output, 'metadata.json'))
        os.remove(os.path.join(output, 'shuffle_policy_train.npz'))
        os.remove(os.path.join(output, 'shuffle_policy_validate.npz'))
        os.remove(os.path.join(output, 'shuffle_policy_test.npz'))
        os.remove(os.path.join(output, 'weights', 'weights.00000.hdf5'))
        os.remove(os.path.join(output, 'weights', 'weights.00001.hdf5'))
        os.rmdir(os.path.join(output, 'weights'))
        os.rmdir(output)

    def testStateAmount(self):
        output = 'tests/test_data/.tmp.state-amount/'
        # create folder
        if not os.path.exists(output):
            os.makedirs(output)

        # shuffle file locations for train/validation/test set
        shuffle_file_train = os.path.join(output, "shuffle_policy_train.npz")
        shuffle_file_val = os.path.join(output, "shuffle_policy_validate.npz")
        shuffle_file_test = os.path.join(output, "shuffle_policy_test.npz")

        # create shuffle files
        create_and_save_shuffle_indices([.9, .05, .05], 1e9,
                                        1033, shuffle_file_train,
                                        shuffle_file_val, shuffle_file_test)

        # load from .npz files
        # load training set
        train_indices = load_indices_from_file(shuffle_file_train)
        # count unique rows
        unique_indices = len(np.vstack({tuple(row) for row in train_indices}))
        self.assertTrue(unique_indices == 7438)

        # load validation set
        val_indices = load_indices_from_file(shuffle_file_val)
        # count unique rows
        unique_indices = len(np.vstack({tuple(row) for row in val_indices}))
        self.assertTrue(unique_indices == 413)

        # load training set
        test_indices = load_indices_from_file(shuffle_file_test)
        # count unique rows
        unique_indices = len(np.vstack({tuple(row) for row in test_indices}))
        self.assertTrue(unique_indices == 413)

        # combine all set rows
        combined_indices = np.concatenate((train_indices, val_indices, test_indices), axis=0)
        # count unique rows
        unique_indices = len(np.vstack({tuple(row) for row in combined_indices}))
        self.assertTrue(unique_indices == (7438 + 413 + 413))

        # remove temporary files
        os.remove(shuffle_file_train)
        os.remove(shuffle_file_val)
        os.remove(shuffle_file_test)
        os.rmdir(output)


class TestBatchGenerator(unittest.TestCase):

    def setUp(self):
        # Create a 2x3x2x2 array to mimic a dataset with shape (batch x features x size x size).
        # Equivalent to np.reshape(np.arange(2*3*2*2), (2, 3, 2, 2))
        self.data = np.array([[[[0, 1], [2, 3]],
                               [[4, 5], [6, 7]],
                               [[8, 9], [10, 11]]],
                              [[[12, 13], [14, 15]],
                               [[16, 17], [18, 19]],
                               [[20, 21], [22, 23]]]])
        # Create (idx, transform) indices in order.
        self.indices = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7],
                                 [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7]])

    def testSymmetries(self):
        # Manually construct expected transformations.
        expected_transforms = {
            "noop": self.data.copy(),
            "rot90": np.array([[[[0, 1], [2, 3]],
                                [[4, 5], [6, 7]],
                                [[8, 9], [10, 11]]],
                               [[[12, 13], [14, 15]],
                                [[16, 17], [18, 19]],
                                [[20, 21], [22, 23]]]])
        }

    def testReshuffle(self):
        # Get expected permutation
        seed = 123
        reshuffle_permutation = np.random.shuffle(np.arange(2))

    def testRestart(Self):
        # Start with initial_batch larger than the size of an epoch (2). Expected behavior is that
        # results are identical - even with reshuffling - to starting from zero and iterating
        # through the same indices.


if __name__ == '__main__':
    unittest.main()
