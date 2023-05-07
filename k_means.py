import numpy as np
import matplotlib.pyplot as plt

class kmeans:
    def __init__(self, k : int) -> None:
        """
        Args:
        -----
            k (int): number of clusters
        """
        self.k = k
        self.labels = None
        self.centroids = None

    def fit(self, data : np.ndarray, epsilon=1e-5) -> np.ndarray:
        """
        fits the centroids and returns the
        labels of the given data

        Args:
        -----
            data (np.ndarray): input data

        Returns:
        --------
            np.ndarray: labels
        """
        n,m = data.shape
        centroids = np.zeros((self.k,m))
        error = 1e9

        
        # setting the first centroid to a random data point
        centroids[0] = data[np.random.randint(0,n)]

        # initializing distances to 0.
        # distances is basically an array whose
        # entries are basically sum of distances
        # from each centroid that has been selected so far

        distances = np.ones(n)
        for i in range(1, self.k):
            # summing distances 
            distances *= np.linalg.norm(data - centroids[i-1], axis=1)
            centroids[i] = data[np.random.choice(np.arange(n), p = distances/np.sum(distances))]
        # ---------------------------------
        # We have initiazed the k centroids


        while error > epsilon:
            # distances are just distance of point to each centorid
            distances = np.zeros(self.k)
            # labels are labels of data points
            labels = np.zeros(n)
            for i,pt in enumerate(data):
                distances = np.linalg.norm(centroids - pt, axis = 1)
                labels[i] = np.argmin(distances)
                
                            
            new_centroids = np.zeros_like(centroids)

            for i in range(self.k):
                new_centroids[i] = np.mean(data[labels == i], axis = 0)
            error = np.linalg.norm(centroids - new_centroids)
            centroids = new_centroids
            # print(error)
            
        self.labels = labels
        self.centroids = centroids
        return labels.astype(int)
    
    

if __name__ == '__main__':
    pts = lambda x : np.random.multivariate_normal(x, 0.01*np.eye(2), size = 100)
    data = np.array([*pts([0,0]), *pts([0,1]), *pts([1,0]), *pts([1,1])])

    clf = kmeans(4)
    labels = clf.fit(data)

    colors = ['white', 'yellow', 'cyan', 'pink']
    plt.style.use('dark_background')
    for i in range(4):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], color=colors[i])
    plt.scatter(clf.centroids[:,0], clf.centroids[:,1] , s = 100, color=colors)
    plt.show()