import argparse

import torch
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
from python_tsp.exact import solve_tsp_dynamic_programming

def compute_center(img_embs, mthd):
    """
    Compute the clustering centers of the input embeddings
    :param img_embs: torch.Tensor, the image embeddings
    :param mthd: the clustering method
    :return: torch.Tensor, the found centers
    """
    mthd = mthd.fit(img_embs)
    centers = mthd.cluster_centers_
    return centers


def compute_route(img_embs, centers):
    dist_matrix = distance_matrix(centers, centers, p=2)
    print(dist_matrix.shape)
    permutation, distance = solve_tsp_dynamic_programming(dist_matrix)

    return permutation


def assign_order(img_embs, route):
    pass


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path', type=str, help='Path to the directory that saves part images')
    args = parser.parse_args()

    # Load in image embeddings
    img_embs = torch.load(args.embedding_path).detach().numpy()

    # Calculate embedding centers
    kmeans = KMeans(n_clusters=25, random_state=0, n_init="auto")
    centers = compute_center(img_embs, kmeans)

    route = compute_route(img_embs, centers)
    print(route)
    # orders = assign_order(img_embs, route)