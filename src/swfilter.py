import matplotlib.pylab as pl
import numpy as np
import copy
from joblib import Parallel, delayed
import ot
from sklearn.metrics.pairwise import euclidean_distances


def collect_sws( data:np.ndarray, n_samples:int, n_dimensions:int, index:int, n:int, eps:float, n_projections:int, seed:int, swtype:str='original'):
        """ Private function used to compute the sliced wasserstein distances for n randomly selected samples. This function is to be parallelized.

        Args:
            data (np.ndarray): The dataset to be analyzed.
            eps (float): The threshold for the outlier detection under the sliced Wasserstein distance.
            n_samples (int): The number of samples in the dataset.
            n_dimensions (int): The number of dimensions in the dataset.
            index (int): The index of the sample to be labeled or not as outlier.
            n (int): The number of samples to be used in the voting for the outlier detection.
            n_projections (int): The number of projections to be used in the computations of the sliced Wasserstein distance.
            seed (int, optional): The random seed used. Defaults to 42.
            swtype (str, optional): The variation of sliced Wasserstein distance used: ['original', 'spherical']. Defaults to 'original'.

        Returns:
            float: The percentage of voters that labeled the sample as an outlier.
        """
        rng = np.random.default_rng(seed=seed)

        a = np.arange(0, n_samples, step=1)
        index_filter =  a != index

        population_size = len(a[index_filter])
        if n >= population_size:
        # Option 1: Adjust n to the population size
            n = population_size
        n_index = rng.choice(a=a[index_filter], size=n, replace=False)
        sws = np.zeros(n)

        data_minus_index = np.delete(data, index, axis=0)
        number = data_minus_index.shape[0]
        a, b = np.ones((number,)) / number, np.ones((number,)) / number  # uniform distribution on samples
        if swtype == 'spherical':
            data_minus_index = data_minus_index/ np.sqrt(np.sum(data_minus_index**2, -1, keepdims=True))

        for i in range(n):
            data_minus_rand_sample = np.delete(data, n_index[i], axis=0)
            if swtype == 'spherical':
                data_minus_rand_sample = data_minus_rand_sample / np.sqrt(np.sum(data_minus_rand_sample**2, -1, keepdims=True))
            sws[i] =  ot.sliced_wasserstein_sphere(data_minus_index, data_minus_rand_sample, a, b, n_projections, seed=seed) if swtype == 'spherical' else  ot.sliced_wasserstein_distance(data_minus_index, data_minus_rand_sample, a = a, b=b, n_projections=n_projections, seed=seed)
        return np.mean(sws >= eps)
    
class SlicedWassersteinFilter:
    """
    A simple outlier detector based on the sliced Wasserstein distance. The function fit_predict use a voting system to label the samples as outliers. 
    A vote entry is obtained by computing the sliced Wasserstein distance between the distribution minus a to be labeled sample and the distribution minus a randomly 
    selected samples from the dataset. The number of randomly selected samples for voting is given by the parameter n. The sample is labeled as an outlier if the percentage of votes 
    is greater than the parameter p. Choose between the original sliced wasserstein distance or its spherical counterpart. This function can be parallelized with the parameter n_jobs.

    Args:   eps (float): The threshold for the outlier detection under the sliced Wasserstein distance.
            n (int): The number of samples to be used in the voting for the outlier detection.
            n_projections (int): The number of projections to be used in the computations of the sliced Wasserstein distance.
            p (float, optional): The percentage of positive votes required to label a sample as an outlier. Defaults to 0.75.
            seed (int, optional): The random seed used. Defaults to 42.
            n_jobs (int, optional): The number of jobs to parallelize the procedure. Defaults to 1.
            swtype (str, optional): The variation of sliced Wasserstein distance used: ['original', 'spherical']. Defaults to 'original'.

    """

    def __init__(self, eps:float, n:int, n_projections:int, p:float = 0.75, seed:int=42, n_jobs:int=1, swtype:str='original') -> None:
        if p >= 1.0:
            raise Exception("p must be lower than 1.0")
        self.eps = eps
        self.n = n
        self.n_projections = n_projections
        self.p = p
        self.seed = seed
        self.n_jobs = n_jobs
        self.swtype = swtype
        self.y_pred = None

    def fit_predict(self, X:np.ndarray, y = None):
        """A simple outlier detector based on the sliced Wasserstein distance. This function use a voting system to label the samples as outliers. 
        A vote entry is obtained by computing the sliced Wasserstein distance between the distribution minus a to be labeled sample and the distribution minus a randomly 
        selected samples from the dataset. The number of randomly selected samples for voting is given by the parameter n. The sample is labeled as an outlier if the percentage of votes 
        is greater than the parameter p. Choose between the original sliced wasserstein distance or its spherical counterpart. This function can be parallelized with the parameter n_jobs.

        Args:
            X (np.ndarray): The dataset to be analyzed.
            y None: Not used, for api calls only.
            
        Returns:
            ndarray: The array with outliers labeled as True.

        """
        
        n_samples = X.shape[0]
        n_dimensions = X.shape[1]
        vote = np.zeros(n_samples)

        results = Parallel(n_jobs=self.n_jobs)(delayed(collect_sws)(X, n_samples=n_samples, n_dimensions=n_dimensions, index=j, n=self.n, eps = self.eps, n_projections=self.n_projections, seed=self.seed, swtype=self.swtype ) for j in range(n_samples))
        vote = np.array(results)
        
        y_pred = (vote >= self.p)
        self.y_pred = np.where(y_pred, -1, 1)
        return self.y_pred, vote
    
    def fit(self, X:np.ndarray, y=None):
        """
        A functions used solely for api calls with sklearn. This function needs to be called before predict and is simply calling fit_predict.
        """
        pass

    def predict(self, X:np.ndarray, y=None):
        """A function used solely for api calls with sklearn. This function needs to be called after fit."""

        return self.fit_predict(X)
    
    def get_params(self, deep=False):
        # Return a dictionary of all parameters
        return {'eps': self.eps, 'n': self.n, 'seed': self.seed, 'swtype': self.swtype, 'n_projections': self.n_projections, 'p': self.p, 'n_jobs': self.n_jobs}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self






def fast_sws_approx( data:np.ndarray, n_samples:int, n_dimensions:int, index:int, n:int, eps:float, seed:int):
        """ Private function used to compute the sliced wasserstein distances for n randomly selected samples. This function is to be parallelized.

        Args:
            data (np.ndarray): The dataset to be analyzed.
            eps (float): The threshold for the outlier detection under the sliced Wasserstein distance.
            n_samples (int): The number of samples in the dataset.
            n_dimensions (int): The number of dimensions in the dataset.
            index (int): The index of the sample to be labeled or not as outlier.
            n (int): The number of samples to be used in the voting for the outlier detection.
            n_projections (int): The number of projections to be used in the computations of the sliced Wasserstein distance.
            seed (int, optional): The random seed used. Defaults to 42.
            swtype (str, optional): The variation of sliced Wasserstein distance used: ['original', 'spherical']. Defaults to 'original'.

        Returns:
            float: The percentage of voters that labeled the sample as an outlier.
        """
        rng = np.random.default_rng(seed=seed)

        a = np.arange(0, n_samples, step=1)
        index_filter =  a != index

        population_size = len(a[index_filter])
        if n >= population_size:
        # Adjust n to the population size
            n = population_size
        n_index = rng.choice(a=a[index_filter], size=n, replace=False)
        sws = np.zeros(n)

        #data_minus_index = np.delete(data, index, axis=0)
        #number = data_minus_index.shape[0]
    
        
        for i in range(n):
            #data_minus_rand_sample = np.delete(data, n_index[i], axis=0)
            sws[i] = np.linalg.norm(data[index,:] - data[n_index[i],:]) #ot.sliced_wasserstein_sphere(data_minus_index, data_minus_rand_sample, a, b, n_projections, seed=seed) if swtype == 'spherical' else  ot.sliced_wasserstein_distance(data_minus_index, data_minus_rand_sample, a = a, b=b, n_projections=n_projections, seed=seed)
        return np.mean(sws >= eps)



class FastSlicedWassersteinFilter:
    """
    A simple outlier detector based on the sliced Wasserstein distance. The function fit_predict use a voting system to label the samples as outliers. 
    A vote entry is obtained by computing the sliced Wasserstein distance between the distribution minus a to be labeled sample and the distribution minus a randomly 
    selected samples from the dataset. The number of randomly selected samples for voting is given by the parameter n. The sample is labeled as an outlier if the percentage of votes 
    is greater than the parameter p. Choose between the original sliced wasserstein distance or its spherical counterpart. This function can be parallelized with the parameter n_jobs.

    Args:   eps (float): The threshold for the outlier detection under the sliced Wasserstein distance.
            n (int): The number of samples to be used in the voting for the outlier detection.
            n_projections (int): The number of projections to be used in the computations of the sliced Wasserstein distance.
            p (float, optional): The percentage of positive votes required to label a sample as an outlier. Defaults to 0.75.
            seed (int, optional): The random seed used. Defaults to 42.
            n_jobs (int, optional): The number of jobs to parallelize the procedure. Defaults to 1.
            swtype (str, optional): The variation of sliced Wasserstein distance used: ['original', 'spherical']. Defaults to 'original'.

    """

    def __init__(self, eps:float, n:int, p:float = 0.75, seed:int=42, n_jobs:int=1) -> None:
        if p >= 1.0:
            raise Exception("p must be lower than 1.0")
        self.eps = eps
        self.n = n
        self.p = p
        self.seed = seed
        self.n_jobs = n_jobs
        self.y_pred = None

    def fit_predict(self, X:np.ndarray, y = None):
        """A simple outlier detector based on the sliced Wasserstein distance. This function use a voting system to label the samples as outliers. 
        A vote entry is obtained by computing the sliced Wasserstein distance between the distribution minus a to be labeled sample and the distribution minus a randomly 
        selected samples from the dataset. The number of randomly selected samples for voting is given by the parameter n. The sample is labeled as an outlier if the percentage of votes 
        is greater than the parameter p. Choose between the original sliced wasserstein distance or its spherical counterpart. This function can be parallelized with the parameter n_jobs.

        Args:
            X (np.ndarray): The dataset to be analyzed.
            y None: Not used, for api calls only.
            
        Returns:
            ndarray: The array with outliers labeled as True.

        """
        
        n_samples = X.shape[0]
        n_dimensions = X.shape[1]
        vote = np.zeros(n_samples)

        results = Parallel(n_jobs=self.n_jobs)(delayed(fast_sws_approx)(X, n_samples=n_samples, n_dimensions=n_dimensions, index=j, n=self.n, eps = self.eps, seed=self.seed) for j in range(n_samples))
        vote = np.array(results)
        
        y_pred = (vote >= self.p)
        self.y_pred = np.where(y_pred, -1, 1)
        return self.y_pred, vote
    
    def fit(self, X:np.ndarray, y=None):
        """
        A functions used solely for api calls with sklearn. This function needs to be called before predict and is simply calling fit_predict.
        """
        pass

    def predict(self, X:np.ndarray, y=None):
        """A function used solely for api calls with sklearn. This function needs to be called after fit."""

        return self.fit_predict(X)
    
    def get_params(self, deep=False):
        # Return a dictionary of all parameters
        return {'eps': self.eps, 'n': self.n, 'seed': self.seed, 'swtype': self.swtype, 'p': self.p, 'n_jobs': self.n_jobs}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self