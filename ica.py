import numpy as np
from numpy.matlib import repmat

class FastICA:
    """
    This class is used to implement the FastIca algorithm,
    as per (A. Hyvarinen and Oja, 2000). In this implementation,
    we optimize the model based on approximations of negentropy.
    """
    def __init__(self, n_i, X, a1=1):
        """ FastICA object constructor. """
        self.W = np.random.rand(n_i, n_i) # initialise unmixing matrix
        self.n_i = n_i
        self.X = X
        self.a1 = a1
        self.avg = None

    def whiten(self, X):
        """ Decorrelates input values.
        Uses the eigenvalue decomposition of the covariance matrix
        to decorrelate the input variables, carrying out a whitening procedure.
        The input variable realization must be centered.
        Args:
            X (ndarray): input mixture data such that every row contains all samples
            of a single variable, and every column contains one sample of each variable.
        Returns:
            ndarray: Output mixture data after whitening procedure.
        """
        # Get eigendecomposition of covariance matrix
        d, E = np.linalg.eig(np.cov(X))
        D_root = np.diag(d ** -0.5) # make diagonal matrix from eigenvalues
        X = E @ D_root @ E.T @ X # whitening step
        self.E, self.D_root = E, D_root # saving for obtaining original unmixing matrix
        return X

    def center(self, X):
        """ Centers data.
        Centers input data by subtracting the sample mean of each variable.
        Args:
            X (ndarray): input mixture data such that every row contains all samples
            of a single variable, and every column contains one sample of each variable.
        Returns:
            ndarray: Output mixture data after centering.
        """
        self.avg = X.mean(axis=1)
        X = (X - repmat(self.avg, X.shape[1], 1).T) # subtract mean
        return X

    def preprocess(self, X):
        """ Preprocesses data.
        Applies the centering process, then the whitening procedure, which are the required
        preprocessing steps prior to performing ICA.
        """
        X = self.center(X) # center X
        X = self.whiten(X) # whiten X
        return X

    def gram_deflate(self, w, W):
        """ Decorrelates the given unmixing vector from the previously found vectors.
        Applies a Gram-Schmidt orthonormalization-like scheme that decorrelates the 
        given unmixing vector, w, from the previously found orthogonal vectors in W,
        preventing the iterative scheme from finding the same independent components
        as in the previous iterations.
        Args:
            w (ndarray): vector containing unmixing vector estimate for a given 
            independent component.
            W (ndarray): matrix containing previously found (and orthonormalized)
            unmixing vectors as rows.
        Returns:
            (ndarray): vector containing unmixing vector estimate after orthonormalization
            procedure.
        """
        # Still more columns to subtract projections from
        w -= np.dot(W[0, :], w)*W[0, :] # subtract projection
        # No more columns to subtract from (base case)
        if W.shape[0] == 1: 
            return w
        else:   
            return self.gram_deflate(w, W[1:, :]) # carry out operation until all projections are subtracted

    def optimize(self, thrs=1e-8, max_itrs=1e3):
        """ Applies FastICA optimization procedure to obtain unmixing matrix estimate.
        Carries out the appropriate preprocessing, Newton estimate updates, and decorrelation
        procedures.
        Args:
            thrs (float): the threshold for deviation from orthogonality in the estimate of a
            given row of the unmixing matrix, used as a convergence criteria.
            max_itrs (int): the maximum desired number of iterations for each row of the unmixing
            matrix, used as stopping criteria. 
        """
        self.X = self.preprocess(self.X)
        n_samples = self.X.shape[1]
        for idx in range(self.n_i): # for each independent component
            converged = False
            itrs = 0
            while not converged:
                w_old =  self.W[idx, :].copy() # variable to track w before update
                # Finding the expected value estimates for the updates
                left_term, right_term = 0, 0
                for n in range(n_samples):
                    left_term += self.g(np.dot(self.W[idx, :], self.X[:, n])) * self.X[:, n]  # term using gradient of measuring function
                    right_term += self.g_deriv(np.dot(self.W[idx, :], self.X[:, n])) # second derivative of measuring function term
                right_term = (right_term/n_samples)*self.W[idx, :] # finding sample mean and multiplying by weights
                left_term = left_term/n_samples # completing sample mean for the left term
                w_plus = left_term - right_term
                # If at least one independent component has been found, apply deflation scheme
                self.W[idx, :] = w_plus
                if idx > 0:
                    self.W[idx, :] = self.gram_deflate(self.W[idx, :], self.W[0:idx, :])
                self.W[idx, :] /= np.linalg.norm(self.W[idx, :])
                # Convergence criteria
                if abs(1 - np.dot(self.W[idx, :], w_old)) < thrs or itrs >= max_itrs:
                    converged = True # converged, end iteration
                else:
                    itrs += 1
    def g(self, x):
        """ First-derivative of contrast function used in negentropy approximation."""
        return np.tanh(self.a1*x)

    def g_deriv(self, x):
        """ Second-derivative of constrast function used in negentropy approximation. """
        return self.a1*(1 - np.tanh(self.a1*x)**2)

    def get_unwhitened_unmixer(self):
        """ Obtains unwhitenened unmixing matrix.
        Due to whitening procedure, the estimated mixing matrix must be "whitened" again
        for comparisons with the original unmixing matrix.
        Returns:
            (ndarray): Re-whitened version of estimated unmixing matrix.
        """
        return self.E @ self.D_root @ self.E.T @ self.W

    def separate_sources(self, X):
        """ Returns independent components present within the source mixtures, X.
        Applies appropriate preprocessing to X and returns independent components
        by multiplying mixtures by unmixing matrix.
        Args:
            X (ndarray): input mixture data such that every row contains all samples
            of a single variable, and every column contains one sample of each variable.
        Returns:
            (ndarray): Output estimated independent components with the same shape as X
            with zero mean and unit variance.
        """
        X = self.preprocess(X)
        S = self.W @ X # separate sources
        return S
    