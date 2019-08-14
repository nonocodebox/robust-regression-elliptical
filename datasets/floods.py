from .dataset import LabeledDataset
import pickle
import numpy as np
import util
from enum import Enum


class FloodsDataset(LabeledDataset):
    class Mode(Enum):
        """
        Feature mode to use.
        """
        D_TO_D = 1  # D_0 -> D_1, D_2, D_3
        DP_TO_DP = 2  # D_0, P_0 -> D_1, P_1, D_2, P_2, D_3, P_3
        DP_TO_D = 3  # D_0, P_0 -> D_1, D_2, D_3

    class StructureType(Enum):
        """
        Strucutre type.
        """
        TIME = 1
        TIMESPACE = 2
        FULL = 3

    def __init__(self, Ns, M, test_size=200, mode=Mode.DP_TO_D, **kwargs):
        super().__init__(**kwargs)

        self.Ns = Ns
        self.M = M
        self.test_size = test_size

        self.mode = mode

        self._load_floods_data()
        self._build_structure()

    def _load_floods_data(self):
        with open('floods.pickle', 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        P = data['precipitation']
        D = data['discharge']

        # Normalize precipitation
        P -= np.mean(P, axis=1, keepdims=True)
        P /= np.std(P, axis=1, keepdims=True)

        # Normalize discharge
        D -= np.mean(D, axis=1, keepdims=True)
        D /= np.std(D, axis=1, keepdims=True)

        num_sites, num_samples = P.shape

        self.P = P
        self.D = D
        self.num_sites = num_sites
        self.num_samples = num_samples

    def _generate_banded(self, n, bands=3):
        """
        Generates a banded matrix.
        :param n: Size of matrix.
        :param bands: Number of bands.
        :return: The banded matrix.
        """
        if bands % 2 == 0:
            raise ValueError('Invalid number of bands')

        B = np.diag(np.ones(n))

        for k in range(1, (bands + 1) // 2):
            v = np.ones(n - k)
            B += np.diag(v, k) + np.diag(v, -k)

        return B

    def _build_structure(self):
        """
        Builds the various structures.
        """

        # Generate the structure for a single site
        if self.mode == self.Mode.D_TO_D:  # D-DDD
            GT = self._generate_banded(4)
        elif self.mode == self.Mode.DP_TO_DP:  # DP-DPDPDP
            GT = self._generate_banded(8, 5)
        elif self.mode == self.Mode.DP_TO_D:  # DP-DDD
            GT = self._generate_banded(5)
            GT[0, :] = 1
            GT[:, 0] = 1
        else:
            raise Exception('Unknown precipitation mode')

        GS_chained = self._generate_banded(self.num_sites)
        GS_isolated = np.identity(self.num_sites)
        E_timespace = util.data.generate_inverse_covariance_structure(np.kron(GT, GS_chained))
        E_time = util.data.generate_inverse_covariance_structure(np.kron(GT, GS_isolated))
        E_full = util.data.generate_inverse_covariance_structure(np.ones((self.num_sites * GT.shape[0], self.num_sites * GT.shape[0])))

        self.Es = {
            self.StructureType.TIMESPACE: E_timespace,
            self.StructureType.TIME: E_time,
            self.StructureType.FULL: E_full
        }

        if self.mode == self.Mode.DP_TO_D:
            self.sample_size = self.num_sites * 2
            self.DD = np.vstack([
                self.P[:, 0:-3],
                self.D[:, 0:-3],
                self.D[:, 1:-2],
                self.D[:, 2:-1],
                self.D[:, 3:]
            ])
        elif self.mode == self.Mode.DP_TO_DP:
            self.sample_size = self.num_sites * 2
            self.DD = np.vstack([
                self.D[:, 0:-3],
                self.P[:, 0:-3],
                self.D[:, 1:-2],
                self.P[:, 1:-2],
                self.D[:, 2:-1],
                self.P[:, 2:-1],
                self.D[:, 3:],
                self.P[:, 3:]
            ])
        elif self.mode == self.Mode.D_TO_D:
            self.sample_size = self.num_sites
            self.DD = np.vstack([
                self.D[:, 0:-3],
                self.D[:, 1:-2],
                self.D[:, 2:-1],
                self.D[:, 3:]
            ])

        self.xy_size = self.DD.shape[0]
        self.target_size = self.xy_size - self.sample_size
        self.num_samples = self.DD.shape[1]

    def get_dimension_x(self):
        return self.sample_size

    def get_dimension_y(self):
        return self.target_size

    def get_Ns(self):
        return self.Ns

    def get_averaging(self):
        return self.M

    def get_edges(self, N_index, iteration):
        raise Exception('Please use a structured version of this dataset')

    def get_train_set(self, N_index, iteration):
        np.random.seed(iteration)
        indices = np.random.permutation(self.num_samples)

        N = self.Ns[N_index]
        return self.DD[:, indices[:N]]

    def get_test_set(self, N_index, iteration):
        np.random.seed(iteration)
        indices = np.random.permutation(self.num_samples)

        if self.test_size == 0:
            N = self.Ns[N_index]
            test_indices = indices[N:]
        else:
            test_indices = indices[-self.test_size:]

        return self.DD[:, test_indices]

    def structured_time(self):
        return self.Structured(self, self.Es[self.StructureType.TIME])

    def structured_timespace(self):
        return self.Structured(self, self.Es[self.StructureType.TIMESPACE])

    def structured_full(self):
        return self.Structured(self, self.Es[self.StructureType.FULL])

    class Structured(LabeledDataset):
        def __init__(self, parent, E, **kwargs):
            super().__init__(**kwargs)

            self.parent = parent
            self.E = E

        def get_dimension_x(self):
            return self.parent.get_dimension_x()

        def get_dimension_y(self):
            return self.parent.get_dimension_y()

        def get_Ns(self):
            return self.parent.get_Ns()

        def get_averaging(self):
            return self.parent.get_averaging()

        def get_edges(self, N_index, iteration):
            return self.E

        def get_train_set(self, N_index, iteration):
            return self.parent.get_train_set(N_index, iteration)

        def get_test_set(self, N_index, iteration):
            return self.parent.get_test_set(N_index, iteration)
