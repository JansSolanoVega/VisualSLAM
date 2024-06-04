import numpy as np


def inlier_detect(pointcloud1, pointcloud2, threshold=0.2):
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

    W, pt_idx_with_max_degree = create_adjacency_matrix_find_node_max_degree(
        pointcloud1, pointcloud2, threshold
    )
    clique = [pt_idx_with_max_degree]
    while True:
        potential_nodes = find_potential_nodes_connected_within_clique(W, clique)
        pt_idx_with_max_degree, max_count = find_potential_node_with_max_degree(
            potential_nodes, W
        )
        if max_count == 0:
            break
        clique.append(pt_idx_with_max_degree)
        if len(clique) > 100:
            break

    return clique


def inlier_detect_iteration(pointcloud1, pointcloud2, threshold=0.2):
    dist_threshold = threshold

    lClique = 0
    clique = []
    while lClique < 6 and len(pointcloud1) >= 6:
        clique = inlier_detect(pointcloud1, pointcloud2, threshold=dist_threshold)
        lClique = len(clique)
        dist_threshold *= 2
        # print(lClique)

    return clique


def create_adjacency_matrix_find_node_max_degree(pc1, pc2, thresh):
    num_points = len(pc1)
    W = np.zeros((num_points, num_points))

    count = 0
    maxn = 0
    maxc = 0

    # diff of pairwise euclidean distance between same points in pc1 and pc2
    for i in range(num_points):
        T1Diff = pc1[i, :] - pc1
        T2Diff = pc2[i, :] - pc2
        T1Dist = np.linalg.norm(T1Diff, axis=1)
        T2Dist = np.linalg.norm(T2Diff, axis=1)
        absDiff = abs(T2Dist - T1Dist)
        wIdx = np.where(absDiff < thresh)
        W[i, wIdx] = 1
        # Find node with max degree
        count = np.sum(W[i, :])
        if count > maxc:
            maxc = count
            maxn = i
        count = 0
    return W, maxn


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
