import numpy as np


class inlier_detector:
    """We assume that the scene is rigid, and hence it must not change between the time instance t and t+1.
    As a result, the distance between any two features in the point cloud Wt must be same as the distance between the corresponding points in Wt+1.
    If any such distance is not same, then either there is an error in 3D triangulation of at least one of the two features, or we have triangulated a moving, which we cannot use in the next step.
    -->From the original point clouds, we now wish to select the largest subset such that they are all the points in this subset are consistent with each other

    Returns:
    The largest subset of pointcloud 1 and pointcloud2, such that they are all the points in this subset are consistent with each other.

    Params:
        -pointcloud1, pointcloud2
        -threshold: If points distances differences is less than 'threshold', the distances are the 'same'
    """

    def __init__(self, pointcloud1, pointcloud2, threshold=0.01):
        self.pc1 = pointcloud1
        self.pc2 = pointcloud2
        self.thresh = threshold

        self.num_points = len(self.pc1)

        self.detect()

    def detect(self):
        W = self.create_adjacency_matrix()
        pt_idx_with_max_degree = self.find_node_with_max_degree(W)
        clique = [pt_idx_with_max_degree]
        while True:
            potential_nodes = self.find_potential_nodes_connected_within_clique(
                W, clique
            )
            pt_idx_with_max_degree, max_count = (
                self.find_potential_node_with_max_degree(potential_nodes, W)
            )
            if max_count == 0:
                break
            clique.append(pt_idx_with_max_degree)
        return self.pc1[clique], self.pc2[clique]

    def create_adjacency_matrix(self):
        W = np.zeros((self.num_points, self.num_points))
        # diff of pairwise euclidean distance between same points in pc1 and pc2
        for i in range(self.num_points):
            for j in range(self.num_points):
                T2Dist = np.linalg.norm(self.pc2[i, :] - self.pc2[j, :])
                T1Dist = np.linalg.norm(self.pc1[i, :] - self.pc1[j, :])
                if abs(T2Dist - T1Dist) < self.thresh:
                    W[i, j] = 1
        return W

    def find_node_with_max_degree(self, W):
        count = 0
        maxn = 0
        maxc = 0

        # Find point with maximum degree and store in maxn
        for i in range(self.num_points):
            for j in range(self.num_points):
                if W[i, j] == 1:
                    count = count + 1
            if count > maxc:
                maxc = count
                maxn = i
            count = 0
        return maxn

    def find_potential_nodes_connected_within_clique(self, W, clique):
        potentialnodes = list()
        isin = True

        # Find potential nodes which are connected to all nodes in the clique
        for i in range(self.num_points):
            for j in range(len(clique)):
                isin = isin & bool(W[i, clique[j]])
            if isin == True and i not in clique:
                potentialnodes.append(i)

        return potentialnodes

    def find_potential_node_with_max_degree(self, potentialnodes, W):
        count = 0
        maxn = 0
        maxc = 0
        # Find the node which is connected to the maximum number of potential nodes and store in maxn
        for i in range(len(potentialnodes)):
            for j in range(len(potentialnodes)):
                if W[potentialnodes[i], potentialnodes[j]] == 1:
                    count = count + 1
            if count > maxc:
                maxc = count
                maxn = potentialnodes[i]
            count = 0
        return maxn, maxc
