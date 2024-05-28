import numpy as np


def inlier_detect(pointcloud1, pointcloud2, threshold=0.01):
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

    W = create_adjacency_matrix(pointcloud1, pointcloud2, threshold)
    pt_idx_with_max_degree = find_node_with_max_degree(W)
    clique = [pt_idx_with_max_degree]
    while True:
        potential_nodes = find_potential_nodes_connected_within_clique(W, clique)
        pt_idx_with_max_degree, max_count = find_potential_node_with_max_degree(
            potential_nodes, W
        )
        if max_count == 0:
            break
        clique.append(pt_idx_with_max_degree)
    return clique


def create_adjacency_matrix(pc1, pc2, thresh):
    num_points = len(pc1)
    W = np.zeros((num_points, num_points))
    # diff of pairwise euclidean distance between same points in pc1 and pc2
    for i in range(num_points):
        for j in range(num_points):
            T2Dist = np.linalg.norm(pc2[i, :] - pc2[j, :])
            T1Dist = np.linalg.norm(pc1[i, :] - pc1[j, :])
            if abs(T2Dist - T1Dist) < thresh:
                W[i, j] = 1
    return W


def find_node_with_max_degree(W):
    num_points = W.shape[0]
    count = 0
    maxn = 0
    maxc = 0

    # Find point with maximum degree and store in maxn
    for i in range(num_points):
        for j in range(num_points):
            if W[i, j] == 1:
                count = count + 1
        if count > maxc:
            maxc = count
            maxn = i
        count = 0
    return maxn


def find_potential_nodes_connected_within_clique(W, clique):
    num_points = W.shape[0]
    potentialnodes = list()
    isin = True

    # Find potential nodes which are connected to all nodes in the clique
    for i in range(num_points):
        for j in range(len(clique)):
            isin = isin & bool(W[i, clique[j]])
        if isin == True and i not in clique:
            potentialnodes.append(i)

    return potentialnodes


def find_potential_node_with_max_degree(potentialnodes, W):
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
