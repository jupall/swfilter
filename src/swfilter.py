import matplotlib.pylab as pl
import numpy as np
import copy
from joblib import Parallel, delayed
import ot
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans


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


class SplitSlicedWassersteinFilter(SlicedWassersteinFilter):
    """
    A clustered outlier detector based on the sliced Wasserstein distance. This class extends the functionality of the simple outlier detector by incorporating clustering 
    into the voting system. The function fit_predict uses a voting system to label the samples as outliers. A vote entry is obtained by computing the sliced Wasserstein 
    distance between the distribution minus a to-be-labeled sample and the distribution minus a randomly selected sample from the dataset. The number of randomly selected 
    samples for voting is given by the parameter n. The sample is labeled as an outlier if the percentage of votes is greater than the parameter p. Choose between the 
    original sliced Wasserstein distance or its spherical counterpart. This function can be parallelized with the parameter n_jobs.

    Additionally, the samples are split into a specified number of clusters (n_clusters) before the voting process. This allows for a faster processing time.

    Args:   eps (float): The threshold for the outlier detection under the sliced Wasserstein distance.
            n (int): The number of samples to be used in the voting for the outlier detection.
            n_projections (int): The number of projections to be used in the computations of the sliced Wasserstein distance.
            p (float, optional): The percentage of positive votes required to label a sample as an outlier. Defaults to 0.75.
            seed (int, optional): The random seed used. Defaults to 42.
            n_jobs (int, optional): The number of jobs to parallelize the procedure. Defaults to 1.
            swtype (str, optional): The variation of sliced Wasserstein distance used: ['original', 'spherical']. Defaults to 'original'.
            n_clusters (int, optional): The number of clusters to partition the data into before performing outlier detection. Defaults to 10.
"""
    def __init__(self, eps:float, n:int, n_projections:int, p:float = 0.75, seed:int=42, n_jobs:int=1, swtype:str='original', n_clusters:int=10) :
        # invoking the __init__ of the parent class
        prop_n=int(n/n_clusters)
        prop_eps= (eps*n_clusters)

        super().__init__(eps=prop_eps, n=prop_n, n_projections=n_projections, p=p, seed=seed, n_jobs=n_jobs, swtype=swtype)
        self.n_clusters = n_clusters
    
        

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
        
        np.random.seed(self.seed)
        
        data_length = X.shape[0]

        # Here we create an array of shuffled indices
        shuf_order = np.arange(data_length)
        np.random.shuffle(shuf_order)

        shuffled_data = copy.deepcopy(X)[shuf_order] # Shuffle the original data

        # Create an inverse of the shuffled index array (to reverse the shuffling operation, or to "unshuffle")
        unshuf_order = np.zeros_like(shuf_order)
        unshuf_order[shuf_order] = np.arange(data_length)

        X_list = np.array_split(shuffled_data, self.n_clusters, axis = 0)
        y_pred = None
        for X_elem in X_list:

        
            n_samples = X_elem.shape[0]
            n_dimensions = X_elem.shape[1]
            vote = np.zeros(n_samples)

            results = Parallel(n_jobs=self.n_jobs)(delayed(collect_sws)(X_elem, n_samples=n_samples, n_dimensions=n_dimensions, index=j, n=self.n, eps = self.eps, n_projections=self.n_projections, seed=self.seed, swtype=self.swtype ) for j in range(n_samples))
            vote = np.array(results)
            
            y_pred_elem = (vote >= self.p)
            #print(y_pred_elem.shape)
            y_pred = np.hstack((y_pred, y_pred_elem)) if y_pred is not None else y_pred_elem
        
        #print(y_pred.shape)
        y_pred = np.array(y_pred).flatten()   
        y_pred = y_pred[unshuf_order] # Unshuffle the shuffled data
        self.y_pred = np.where(y_pred, -1, 1)
       
        return self.y_pred, vote
    
    def get_params(self, deep=False):
        # Return a dictionary of all parameters

        return super().get_params() | {'n_clusters': self.n_clusters}
    
class SmartSplitSlicedWassersteinFilter(SlicedWassersteinFilter):
    """
    A clustered outlier detector based on the sliced Wasserstein distance. This class extends the functionality of the simple outlier detector by incorporating clustering 
    into the voting system and a smart sampling method. The function fit_predict uses a voting system to label the samples as outliers. A vote entry is obtained by computing the sliced Wasserstein 
    distance between the distribution minus a to-be-labeled sample and the distribution minus a randomly selected sample from the dataset. The number of randomly selected 
    samples for voting is given by the parameter n. The sample is labeled as an outlier if the percentage of votes is greater than the parameter p. Choose between the 
    original sliced Wasserstein distance or its spherical counterpart. This function can be parallelized with the parameter n_jobs.

    Additionally, the samples are split into a specified number of clusters (n_clusters) before the voting process. This allows for a faster processing time.

    Args:   eps (float): The threshold for the outlier detection under the sliced Wasserstein distance.
            n (int): The number of samples to be used in the voting for the outlier detection.
            n_projections (int): The number of projections to be used in the computations of the sliced Wasserstein distance.
            p (float, optional): The percentage of positive votes required to label a sample as an outlier. Defaults to 0.75.
            seed (int, optional): The random seed used. Defaults to 42.
            n_jobs (int, optional): The number of jobs to parallelize the procedure. Defaults to 1.
            swtype (str, optional): The variation of sliced Wasserstein distance used: ['original', 'spherical']. Defaults to 'original'.
            n_splits (int, optional): The number of split to partition the data into before performing outlier detection. Defaults to 10.
            n_clusters (int, optional): The number of clusters to partition the data into before performing outlier detection. Defaults to 10.
"""
    def __init__(self, eps:float, n:int, n_projections:int, p:float = 0.75, seed:int=42, n_jobs:int=1, swtype:str='original', n_clusters:int=10, n_splits:int=10) :
        # invoking the __init__ of the parent class
        prop_n=int(n/n_splits)
        prop_eps= (eps*n_splits)

        super().__init__(eps=prop_eps, n=prop_n, n_projections=n_projections, p=p, seed=seed, n_jobs=n_jobs, swtype=swtype)
        self.n_clusters = n_clusters
        self.n_splits = n_splits
    
        

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
        
        np.random.seed(self.seed)
        
        data_length = X.shape[0]

        # Here we create an array of shuffled indices
        indices = np.arange(data_length)
        #print(f"ortiginal indices: {indices}")

        # Here we create an array of shuffled indices
        # i) use k means to split the dataset in n_clusters
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init="auto").fit(X)
        # ii) split each cluster into n_splits and take one split of each cluster to form a subset of the data with similar properties to the original dataset
        labels = kmeans.labels_
        #print(f"kmeans labels: {labels}")

        cluster_splits_indices = []
        for i in range(self.n_clusters):
            cluster_indices = indices[labels == i]
            cluster_indices = shuffle(cluster_indices, random_state=self.seed)
            cluster_splits_indices += [np.array_split(cluster_indices, self.n_splits)]
        
        #print(f"cluster_splits_indices: {cluster_splits_indices}")

        X_split_indices = [None] * self.n_splits
        for elem in cluster_splits_indices:
            for i in range(self.n_splits):
               X_split_indices[i] = np.hstack((X_split_indices[i], elem[i])) if X_split_indices[i] is not None else elem[i]

        #print(f"X_split_indices: {X_split_indices}")
        X_list = [X[elem] for elem in X_split_indices]    

        #print(f"X_list: {X_list}")
        y_pred = None
        flat_X_indices = None
        full_vote = None
        # iii) apply the algorithm to each subset
        for X_elem in X_list:

        
            n_samples = X_elem.shape[0]
            n_dimensions = X_elem.shape[1]
            vote = np.zeros(n_samples)

            results = Parallel(n_jobs=self.n_jobs)(delayed(collect_sws)(X_elem, n_samples=n_samples, n_dimensions=n_dimensions, index=j, n=self.n, eps = self.eps, n_projections=self.n_projections, seed=self.seed, swtype=self.swtype ) for j in range(n_samples))
            vote = np.array(results)
            
            y_pred_elem = (vote >= self.p)
            #print(y_pred_elem.shape)
            y_pred = np.hstack((y_pred, y_pred_elem)) if y_pred is not None else y_pred_elem
            full_vote = np.hstack((full_vote, vote)) if full_vote is not None else vote
            
        for elem in X_split_indices:
            flat_X_indices = np.hstack((flat_X_indices, elem)) if flat_X_indices is not None else elem
        # iv) combine the results to form the final result
        #print(f"y_pred.shape: {y_pred.shape}")
        y_pred = np.array(y_pred).flatten()
        flat_X_indices = np.array(flat_X_indices).flatten()
        full_vote = np.array(full_vote).flatten()
        #print(f"flat_X_indices: {flat_X_indices}")


        unshuf_order = np.zeros_like(indices)
        unshuf_order[flat_X_indices] = np.arange(data_length)  
        y_pred = y_pred[unshuf_order] # Unshuffle the shuffled data
        full_vote = full_vote[unshuf_order]
        self.y_pred = np.where(y_pred, -1, 1)
       
        return self.y_pred, full_vote
    
    def get_params(self, deep=False):
        # Return a dictionary of all parameters

        return super().get_params() | {'n_clusters': self.n_clusters}



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


class FastEuclidianFilter:
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
    

"""
def collect_sws_multi( data:np.ndarray, n_samples:int, n_dimensions:int, group_index:np.ndarray, n:int, eps:float, n_projections:int, n_groups:int, seed:int, swtype:str='original'):
    
    #print(f"number of groups: {n_groups}")
    rng = np.random.default_rng(seed=seed)

    a = np.arange(0, n_samples, step=1)
    #print(f"shape of a: {a.shape}")
    #print(f"max of a: {np.max(a)}")
    mask = np.ones(a.shape,dtype=bool)
    #print(f"shape of mask: {mask.shape}")
    mask[group_index] = 0
    index_filter = a[mask]
    #print(f"shape of index_filter: {index_filter.shape}")

    population_size = len(index_filter)
    #print(f"population size: {population_size}")
    if n >= 10*n_samples:
    # Option 1: Adjust n to the population size
        n = 10*n_samples
    #print(f"n: {n}")
    #print(f"group_index.size: {group_index.size}")
    n_index = np.zeros((n, group_index.size), dtype=int)
    for i in range(n):
        rng2 = np.random.default_rng(seed=i)
        n_index[i] = rng2.choice(a=index_filter, size=group_index.size, replace=False, shuffle=True)
    
    #n_index = rng.choice(a=index_filter.flatten(), size=(n,int(group_index.size)), replace=False)
    #print(f"n_index.shape: {n_index.shape}")

    #print(f"data.shape: {data.shape}")
    data_minus_index = np.delete(data, group_index, axis=0)
    #print(f"data_minus_index.shape: {data_minus_index.shape}")

    number = data_minus_index.shape[0]
    #print(f"number: {number}")
    a, b = np.ones((number,)) / number, np.ones((number,)) / number  # uniform distribution on samples
    if swtype == 'spherical':
        data_minus_index = data_minus_index/ np.sqrt(np.sum(data_minus_index**2, -1, keepdims=True))

    dist_map = []
    
    for i in range(n):
        data_minus_rand_samples = np.delete(data, n_index[i,:], axis=0)
        #print(f"data_minus_rand_samples.shape: {data_minus_rand_samples.shape}")
        if swtype == 'spherical':
            data_minus_rand_samples = data_minus_rand_samples / np.sqrt(np.sum(data_minus_rand_samples**2, -1, keepdims=True))
        dist =  ot.sliced_wasserstein_sphere(data_minus_index, data_minus_rand_samples, a, b, n_projections, seed=seed) if swtype == 'spherical' else  ot.sliced_wasserstein_distance(data_minus_index, data_minus_rand_samples, a = a, b=b, n_projections=n_projections, seed=seed)
        
        keys =  n_index[i,:] #np.concatenate((group_index, n_index[i,:]))
        dict_dist = dict.fromkeys(keys, None)

        for key in keys:
            dict_dist[key] = dist
    
        dist_map.append(dict_dist)

        
    return dist_map # returns a list of dictionaries with the index of each element and the corresponding sw-distance


class MultiSampleSlicedWassersteinFilter:
   

    def __init__(self, eps:float, n:int, cluster_size:int, n_projections:int, p:float = 0.75, seed:int=42, n_jobs:int=1, swtype:str='original') -> None:
        if p >= 1.0:
            raise Exception("p must be lower than 1.0")
        self.eps = eps
        self.n = n
        self.cluster_size = cluster_size
        self.n_projections = n_projections
        self.p = p
        self.seed = seed
        self.n_jobs = n_jobs
        self.swtype = swtype
        self.y_pred = None

    def fit_predict(self, X:np.ndarray, y = None):
       
        
        n_samples = X.shape[0]
        n_dimensions = X.shape[1]
        vote = np.zeros(n_samples)
        groups_index_list = np.array_split(np.arange(n_samples), int(n_samples/self.cluster_size))
        n_groups = int(len(groups_index_list))
        

        results = Parallel(n_jobs=self.n_jobs)(delayed(collect_sws_multi)(X, n_samples=n_samples, n_dimensions=n_dimensions, group_index=group_index, n=self.n, eps = self.eps, n_projections=self.n_projections, n_groups=n_groups, seed=self.seed, swtype=self.swtype ) for group_index in (groups_index_list))
        
        
        total_res = {}
        for elem in results:
            for elem2 in elem:
                for key in elem2:
                    if key not in total_res:
                        total_res[key] = {'sum':0, 'count':0}
                    total_res[key]['sum'] = elem2[key]
                    total_res[key]['count'] += 1

        mean = np.zeros(n_samples)
        for key in total_res:
            mean[key] = total_res[key]['sum'] / total_res[key]['count']
        
        vote = (mean - np.min(mean)) / (np.max(mean) - np.min(mean))
       
        vote = vote.flatten()
        vote_index = vote.argsort()
        
        last_index = int(self.p * vote.size)
        
        outliers_index = vote_index[last_index:]
        inliers_index = vote_index[:last_index]
        vote[outliers_index] = int(-1)
        
        vote[inliers_index] = int(1)
      
        y_pred = np.array(vote, dtype=int)
        self.y_pred = y_pred
        return y_pred, mean
    
    def fit(self, X:np.ndarray, y=None):
        
        pass

    def predict(self, X:np.ndarray, y=None):
        

        return self.fit_predict(X)
    
    def get_params(self, deep=False):
        # Return a dictionary of all parameters
        return {'eps': self.eps, 'n': self.n, 'seed': self.seed, 'swtype': self.swtype, 'n_projections': self.n_projections, 'p': self.p, 'n_jobs': self.n_jobs}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
"""