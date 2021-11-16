import unittest
from ica import FastICA
import numpy as np

class TestFastICA(unittest.TestCase):
    """ Class used to test FastICA functionality. preprocess() is not tested as it just
    applies standardize() and whiten().
    """
    def setUp(self):
        # Sets up initial test case to be used to test the different methods
        n_samples = int(1e4)
        self.S = np.array([np.random.rand(n_samples), np.random.rand(n_samples)])
        self.W = np.array([[1, 2], [2, 1]]) # mixing matrix
        self.A = np.linalg.inv(self.W)
        self.X = self.A @ self.S # getting our linear mixtures
        self.n_i = 2
        self.model = FastICA(self.n_i, self.X)

    def test_center(self):
        # Tests whether input data has been standardized
        X = self.model.center(self.X)
        avg = X.mean(axis=1)
        self.assertTrue(np.isclose(avg[0], 0) and np.isclose(avg[1], 0))
    
    def test_whiten(self):
        # Tests whether input data best been whitened
        X = self.model.center(self.X)
        X = self.model.whiten(X)
        self.assertTrue(np.allclose(np.cov(X), np.identity(n=self.n_i)))

    def test_gram_deflate(self):
        # Tests whether or not method carries out gram-schmidt orthonormalization correctly
        w = np.array([1, 2, 3])
        w = w / np.linalg.norm(w)
        W = np.array([[1/(3**0.5), 1/(3**0.5), 1/(3**0.5)],
                    [-1/(3**0.5), 1/(3**0.5), -1/(3**0.5)]]) # unit vectors
        w_out = self.model.gram_deflate(w, W)
        self.assertTrue(np.isclose(np.dot(w_out, W[0, :]), 0))
        self.assertTrue(np.isclose(np.dot(w_out, W[1, :]), 0))

    def test_optimize(self):
        # Tests whether optimize method works by comparing obtained unmixing matrix with original
        self.model.optimize()
        n = np.linalg.norm
        W_est = self.model.get_unwhitened_unmixer()
        w1, w1_est = self.W[0, :], W_est[0, :]
        w2, w2_est = self.W[1, :], W_est[1, :]
        self.assertTrue(np.isclose(abs((w1/n(w1)).dot(w1_est/n(w1_est))), 1, atol=0.5))
        self.assertTrue(np.isclose(abs((w1/n(w1)).dot(w1_est/n(w1_est))), 1, atol=0.5))

        

if __name__ == '__main__':
    unittest.main()