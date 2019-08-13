import util


class Dataset(object):
    def get_dimension(self):
        raise NotImplementedError('Method must be overridden in a derived class')

    def get_Ns(self):
        raise NotImplementedError('Method must be overridden in a derived class')

    def get_averaging(self):
        raise NotImplementedError('Method must be overridden in a derived class')

    def get_edges(self, N_index, iteration):
        raise NotImplementedError('Method must be overridden in a derived class')

    def get_train_set(self, N_index, iteration):
        raise NotImplementedError('Method must be overridden in a derived class')

    def get_test_set(self, N_index, iteration):
        raise NotImplementedError('Method must be overridden in a derived class')


class ConditionalDataset(Dataset):
    def get_dimension_x(self):
        raise NotImplementedError('Method must be overridden in a derived class')

    def get_dimension_y(self):
        raise NotImplementedError('Method must be overridden in a derived class')

    def get_dimension(self):
        return self.get_dimension_x() + self.get_dimension_y()

    def get_edges_yy(self, N_index, iteration):
        _, E_yx, E_yy = util.split_edges(self.get_edges(N_index, iteration), self.get_dimension_x(), self.get_dimension_y())

        return E_yy

    def get_edges_yx(self, N_index, iteration):
        _, E_yx, E_yy = util.split_edges(self.get_edges(N_index, iteration), self.get_dimension_x(), self.get_dimension_y())

        return E_yx

    def get_train_set_x(self, N_index, iteration):
        return self.get_train_set(N_index, iteration)[:self.get_dimension_x(), :]

    def get_train_set_y(self, N_index, iteration):
        return self.get_train_set(N_index, iteration)[self.get_dimension_x():, :]

    def get_test_set_x(self, N_index, iteration):
        return self.get_test_set(N_index, iteration)[:self.get_dimension_x(), :]

    def get_test_set_y(self, N_index, iteration):
        return self.get_test_set(N_index, iteration)[self.get_dimension_x():, :]
