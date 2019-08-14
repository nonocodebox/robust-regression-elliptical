import util


class Dataset(object):
    """
    A dataset object which encapsulates the dataset and dataset information.
    """

    def get_dimension(self):
        """
        Returns the total number of features.
        :return: Total number of features.
        """
        raise NotImplementedError('Method must be overridden in a derived class')

    def get_Ns(self):
        """
        Returns a list of training sample sizes.
        :return: List of number of samples.
        """
        raise NotImplementedError('Method must be overridden in a derived class')

    def get_averaging(self):
        """
        Returns the number of times calculated result should be averaged.
        :return: The number of averaging iterations.
        """
        raise NotImplementedError('Method must be overridden in a derived class')

    def get_edges(self, N_index, iteration):
        """
        Returns the dataset's predefined structure.
        :param N_index: The index of the N value in the list returned by get_Ns().
        :param iteration: The iteration number.
        :return: The dataset's predefined structure, which can possibly depend on N_index and iteration.
        """
        raise NotImplementedError('Method must be overridden in a derived class')

    def get_train_set(self, N_index, iteration):
        """
        Returns training data of size (number of features, number of training samples)
        :param N_index: The index of the N value in the list returned by get_Ns().
        :param iteration: The iteration number.
        :return: The train set, depending on N_index and iteration.
        """
        raise NotImplementedError('Method must be overridden in a derived class')

    def get_test_set(self, N_index, iteration):
        """
        Returns test data of size (number of features, number of train samples)
        :param N_index: The index of the N value in the list returned by get_Ns().
        :param iteration: The iteration number.
        :return: The test set, depending on N_index and iteration.
        """
        raise NotImplementedError('Method must be overridden in a derived class')


class LabeledDataset(Dataset):
    """
    A dataset object for labeled data.
    """

    def get_dimension_x(self):
        """
        Returns the number of input features.
        :return: Number of input features.
        """
        raise NotImplementedError('Method must be overridden in a derived class')

    def get_dimension_y(self):
        """
        Returns the number of targets.
        :return: Number of targets.
        """
        raise NotImplementedError('Method must be overridden in a derived class')

    def get_dimension(self):
        """
        Returns the total number of features (input features + targets).
        :return: Total number of features.
        """
        return self.get_dimension_x() + self.get_dimension_y()

    def get_edges_yy(self, N_index, iteration):
        """
        Returns the dataset's predefined targets structure.
        :param N_index: The index of the N value in the list returned by get_Ns().
        :param iteration: The iteration number.
        :return: The dataset's predefined targets structure, which can possibly depend on N_index and iteration.
        """
        _, E_yx, E_yy = util.split_edges(self.get_edges(N_index, iteration), self.get_dimension_x(), self.get_dimension_y())

        return E_yy

    def get_edges_yx(self, N_index, iteration):
        """
        Returns the dataset's predefined targets structure.
        :param N_index: The index of the N value in the list returned by get_Ns().
        :param iteration: The iteration number.
        :return: The dataset's predefined targets-features structure, which can possibly depend on N_index and iteration.
        """
        _, E_yx, E_yy = util.split_edges(self.get_edges(N_index, iteration), self.get_dimension_x(), self.get_dimension_y())

        return E_yx

    def get_train_set_x(self, N_index, iteration):
        """
        Returns input features of the training data of size (number of input features, number of training samples)
        :param N_index: The index of the N value in the list returned by get_Ns().
        :param iteration: The iteration number.
        :return: The input features part of the train set, depending on N_index and iteration.
        """
        return self.get_train_set(N_index, iteration)[:self.get_dimension_x(), :]

    def get_train_set_y(self, N_index, iteration):
        """
        Returns targets of the training data of size (number of targets, number of training samples)
        :param N_index: The index of the N value in the list returned by get_Ns().
        :param iteration: The iteration number.
        :return: The targets part of the train set, depending on N_index and iteration.
        """
        return self.get_train_set(N_index, iteration)[self.get_dimension_x():, :]

    def get_test_set_x(self, N_index, iteration):
        """
        Returns input features of the test data of size (number of input features, number of test samples)
        :param N_index: The index of the N value in the list returned by get_Ns().
        :param iteration: The iteration number.
        :return: The input features part of the test set, depending on N_index and iteration.
        """
        return self.get_test_set(N_index, iteration)[:self.get_dimension_x(), :]

    def get_test_set_y(self, N_index, iteration):
        """
        Returns targets of the test data of size (number of targets, number of test samples)
        :param N_index: The index of the N value in the list returned by get_Ns().
        :param iteration: The iteration number.
        :return: The targets part of the test set, depending on N_index and iteration.
        """
        return self.get_test_set(N_index, iteration)[self.get_dimension_x():, :]
