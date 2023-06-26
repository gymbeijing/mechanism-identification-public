import argparse
import logging
import sys

import torch
from sklearn.cluster import KMeans
import hdbscan
from scipy.spatial import distance_matrix
# from python_tsp.exact import solve_tsp_dynamic_programming
# from python_tsp.heuristics import solve_tsp_local_search as solve_tsp
from python_tsp.heuristics import solve_tsp_simulated_annealing as solve_tsp
import json
from numpy.linalg import norm
import numpy as np
import math
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


NUM_CLUSTERS = 25
REDUCED_DIM = 5
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


def dot(v, w):
    # x, y, z = v
    # X, Y, Z = w
    # return x * X + y * Y + z * Z
    return v.dot(w)


def length(v):
    # x, y, z = v
    # return math.sqrt(x * x + y * y + z * z)
    return norm(v)


def vector(b, e):
    # x, y, z = b
    # X, Y, Z = e
    # return X - x, Y - y, Z - z
    return e - b


def unit(v):
    # x, y, z = v
    # mag = length(v)
    # return x / mag, y / mag, z / mag
    mag = length(v)
    return v / mag


def distance(p0, p1):
    return length(vector(p0, p1))


def scale(v, sc):
    # x, y, z = v
    # return x * sc, y * sc, z * sc
    return v * sc


def add(v, w):
    # x, y, z = v
    # X, Y, Z = w
    # return x + X, y + Y, z + Z
    return v + w


# Given a line with coordinates 'start' and 'end' and the
# coordinates of a point 'pnt' the proc returns the shortest
# distance from pnt to the line and the coordinates of the
# nearest point on the line.
#
# Malcolm Kesson 16 Dec 2012

def pnt2line(pnt, start, end):
    # 1  Convert the line segment to a vector ('line_vec').
    line_vec = vector(start, end)
    # 2  Create a vector connecting start to pnt ('pnt_vec').
    pnt_vec = vector(start, pnt)
    # 3  Find the length of the line vector ('line_len').
    line_len = length(line_vec)
    # 4  Convert line_vec to a unit vector ('line_unitvec').
    line_unitvec = unit(line_vec)
    # 5  Scale pnt_vec by line_len ('pnt_vec_scaled').
    pnt_vec_scaled = scale(pnt_vec, 1.0 / line_len)
    # 6  Get the dot product of line_unitvec and pnt_vec_scaled ('t').
    t = dot(line_unitvec, pnt_vec_scaled)
    # 7  Ensure t is in the range 0 to 1.
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    # 8  Use t to get the nearest location on the line to the end
    #    of vector pnt_vec_scaled ('nearest').
    nearest = scale(line_vec, t)
    # 9  Calculate the distance from nearest to pnt_vec_scaled.
    dist = distance(nearest, pnt_vec)
    # 10 Translate nearest back to the start/end line.
    nearest = add(nearest, start)
    return dist, nearest


def compute_center(img_embs, mthd):
    """
    Compute the clustering centers of the input embeddings
    :param img_embs: torch.Tensor, the image embeddings
    :param mthd: the clustering method
    :return: torch.Tensor, the found centers
    """
    mthd = mthd.fit(img_embs)
    centers = mthd.cluster_centers_
    return centers, mthd


def compute_route(centers):
    """
    Compute the TSP route
    :param centers: the found clustering centers among the embeddings
    :return: a list of the centers (in ordinal number) in the order of TSP route
    """
    dist_matrix = distance_matrix(centers, centers, p=2)
    #print(dist_matrix.shape)
    permutation, distance = solve_tsp(dist_matrix)
    # print(distance)

    return permutation


def assign_order(img_embs, assembly_id_dict, metadata_list, centers, route):
    """
    Assign orders to the parts in each assembly
    :param img_embs: torch.Tensor, all the image embeddings
    :param assembly_id_dict: {assembly_id: indexes of the image embedding that belongs to this assembly}
    :param metadata_list: list of the metadata corresponding to the image
    :param centers: torch.Tensor, computed clustering centers
    :param route: a list of ordinal numbers representing the TSP route found from the centers
    :return: {assembly_id: [ordered_part1_metadata, ordered_part2_meta_data ...]}
    """
    # Compute line segments between two adjacent nodes on the route
    # Use the ordinal number, start and the end node's embedding as the representation of the line segment
    segments = dict()
    for i in range(len(route)):
        if i + 1 < len(route):
            segments[i] = {'start': centers[route[i]], 'end': centers[route[i + 1]]}

    segments[len(route) - 1] = {'start': centers[route[-1]], 'end': centers[route[0]]}

    # Compute the ordering helping information of each image embedding
    emb_order_info_list = []
    for emb in img_embs:
        emb_segment_id, emb_projected_dist = compute_segment_id(emb, segments, route)
        emb_order_info_list.append({'seg_id_order': emb_segment_id, 'proj_dist': emb_projected_dist})

    # print(emb_order_info_list)
    # all_sorted_parts = sorted(enumerate(emb_order_info_list), key=lambda d: (d[1]['seg_id_order'], d[1]['proj_dist']))

    # all_ordered_assembly_parts = [metadata_list[i] for i, _ in all_sorted_parts]
    # with open('../processed_data' + '/' + f'all_part_orders.json', 'w', encoding='utf8') as fp:
    #     json.dump(all_ordered_assembly_parts, fp, indent=4, ensure_ascii=False, sort_keys=False)

    # Compute the orders of the parts for each assembly
    ordered_assembly_parts = dict()
    for assembly_id, indexes in assembly_id_dict.items():
        # print(f"{assembly_id}: {indexes}")
        # print(emb_order_info_list[indexes[0]:indexes[-1]+1])
        emb_order_info_slice = emb_order_info_list[indexes[0]:indexes[-1]+1]
        sorted_parts = sorted(enumerate(emb_order_info_slice), key=lambda d: (d[1]['seg_id_order'], d[1]['proj_dist']))
        ordered_assembly_parts[assembly_id] = [metadata_list[indexes[i]] for i, _ in sorted_parts]

    return ordered_assembly_parts


def compute_segment_id(emb, segments, route):
    """
    Compute the assigned line segment and projected distance of an embedding
    :param emb: the embedding to be assigned line segment
    :param segments: {ordered_segment_id: {'start': <start_node_embedding>, 'end': <end_node_embedding>}}
    :param route: the computed TSP route
    :return: the closest segment's id emb gets assigned to, and the projected distance between the starting node and the
    nearest point
    """
    shortest_dist = 10000000
    closest_seg_id_order = 0
    nearest = np.zeros_like(emb)
    for seg_id, seg in segments.items():
        dist, proj_pnt = pnt2line(emb, seg['start'], seg['end'])
        if dist < shortest_dist:
            shortest_dist = dist
            closest_seg_id_order = route[seg_id]
            nearest = proj_pnt
    # projected_dist = math.sqrt(math.pow(math.dist(emb, segments[closest_seg_id]), 2) + math.pow(shortest_dist, 2))
    projected_dist = math.dist(segments[closest_seg_id_order]['start'], nearest)

    return closest_seg_id_order, projected_dist


def save_orders_to_file(orders, dirname):
    logging.info(f'Saving part orders to {dirname}/part_orders_dim=5.json')
    with open(dirname+'/'+f'part_orders_dim=5.json', 'w', encoding='utf8') as fp:
        json.dump(orders, fp, indent=4, ensure_ascii=False, sort_keys=False)

    return


def compute_closest_emb_to_clusters(centers, img_embs, cluster_labels, metadata):
    # print(cluster_labels)
    closest_img_embs_name = dict()
    cluster_labels = np.array(cluster_labels)
    for c_idx in range(NUM_CLUSTERS):
        indexes = np.where(cluster_labels == c_idx)[0]
        #print(len(indexes))
        shortest_dist = 10000000
        for img_idx in indexes:
            if math.dist(centers[c_idx], img_embs[img_idx]) < shortest_dist:
                closest_img_embs_name[c_idx] = metadata[img_idx]
                shortest_dist = math.dist(centers[c_idx], img_embs[img_idx])

    return closest_img_embs_name


def reduce_dimension(X, d):
    # PCA
    logging.info('Using PCA to reduce dimension...')
    pca = PCA(n_components=d)
    X_new = pca.fit_transform(X)

    # TSNE
    #logging.info('Using TSNE...')
    #tsne = TSNE(n_components=d)
    #X_new = tsne.fit_transform(X_new)
    
    # UMAP	
    #logging.info('Using UMAP...')
    #m = umap.UMAP(n_components=5)
    #X_new = m.fit_transform(X)
    
    return X_new


def save_cluster_info(labels, metadata, dirname):
    cluster_image_name_dict = dict()
    assert len(labels)==len(metadata), "The number of labels is not equal to the number of items in the metadata"
    logging.info(f'Length of labels: {len(labels)}')

    for i, label in enumerate(labels):
        if label not in cluster_image_name_dict:
            cluster_image_name_dict[label.item()] = [metadata[i]]
        else:
            cluster_image_name_dict[label.item()].append(metadata[i])

    logging.info(f'Saving cluster_info to {dirname}/cluster_info_dim=512.json')
    with open(f'{dirname}/cluster_info_dim=512.json', 'w', encoding='utf8') as fp:
        json.dump(cluster_image_name_dict, fp, indent=4, ensure_ascii=False, sort_keys=False)

    return
    


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path', type=str, help='Path to the file that stores part images')
    parser.add_argument('--metadata_path', type=str, help='Path to the file that stores embedding metadata')
    args = parser.parse_args()

    # Load in image embeddings
    #img_embs = torch.load(args.embedding_path).detach().numpy()
    img_embs = torch.load(args.embedding_path).numpy()

    # Dimensionality reduction
    img_embs = reduce_dimension(img_embs, d=REDUCED_DIM)
    logging.info(f"Image embedding dimension reduced to {REDUCED_DIM} by PCA")
    # Calculate embedding centers
    '''    
    for num_clusters in [5, 10, 15, 20, 25, 30, 35, 40]:
        logging.info(f'kmeans with num_clusters={num_clusters}')
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto")
        centers, kmeans = compute_center(img_embs, kmeans)
        cluster_labels = kmeans.labels_
        logging.info(f'Silhouette Score(n={num_clusters}): {silhouette_score(img_embs, cluster_labels)}')
    '''
    ''' 
    for min_cluster_size in [1250, 2500, 3500, 4500]:
        logging.info(f'hdbscan with min_cluster_size={min_cluster_size}')
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        cluster_labels = clusterer.fit_predict(img_embs)
        logging.info(f'Silhouette Score(n={min_cluster_size}): {silhouette_score(img_embs, cluster_labels)}')
    '''

    
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0, n_init="auto")
    centers, kmeans = compute_center(img_embs, kmeans)
    cluster_labels = kmeans.labels_
    

    # Load assembly ids from the metadata
    with open(args.metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Compute the closest img to the cluster
    #closest_img_embs_name = compute_closest_emb_to_clusters(centers, img_embs, cluster_labels, metadata)
    #logging.info(f'Printing the closest image embedding to each cluster...')
    #print(closest_img_embs_name)

    #save_cluster_info(cluster_labels, metadata, '../processed_data')

    #sys.exit()

    # Calculate the TSP route
    route = compute_route(centers)
    logging.info('Printing the route...')
    print(route)
    
    # Compute the dict mapping assembly id to the list of indexes of the part belonging to the assembly
    assembly_id_dict = dict()
    for idx in range(len(metadata)):
        m = metadata[idx]
        m_split = m.split('_')
        assembly_id = f'{m_split[1]}_{m_split[2]}'
        if assembly_id not in assembly_id_dict:
            assembly_id_dict[assembly_id] = []
            assembly_id_dict[assembly_id].append(idx)
        else:
            assembly_id_dict[assembly_id].append(idx)

    # Assign order to the parts in each assembly
    orders = assign_order(img_embs, assembly_id_dict, metadata, centers, route)
    #vid = args.embedding_path.split('/')[-1].split('.')[0].split('_')[-1]
    save_orders_to_file(orders, '../processed_data')
    
