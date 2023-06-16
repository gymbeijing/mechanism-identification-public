import argparse

import torch
from sklearn.cluster import KMeans
from scipy.spatial import distance_matrix
# from python_tsp.exact import solve_tsp_dynamic_programming
# from python_tsp.heuristics import solve_tsp_local_search as solve_tsp
from python_tsp.heuristics import solve_tsp_simulated_annealing as solve_tsp
import json
from numpy.linalg import norm
import numpy as np
import math

import math


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
# 1  Convert the line segment to a vector ('line_vec').
# 2  Create a vector connecting start to pnt ('pnt_vec').
# 3  Find the length of the line vector ('line_len').
# 4  Convert line_vec to a unit vector ('line_unitvec').
# 5  Scale pnt_vec by line_len ('pnt_vec_scaled').
# 6  Get the dot product of line_unitvec and pnt_vec_scaled ('t').
# 7  Ensure t is in the range 0 to 1.
# 8  Use t to get the nearest location on the line to the end
#    of vector pnt_vec_scaled ('nearest').
# 9  Calculate the distance from nearest to pnt_vec_scaled.
# 10 Translate nearest back to the start/end line.
# Malcolm Kesson 16 Dec 2012

def pnt2line(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0 / line_len)
    t = dot(line_unitvec, pnt_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
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
    return centers


def compute_route(centers):
    dist_matrix = distance_matrix(centers, centers, p=2)
    print(dist_matrix.shape)
    permutation, distance = solve_tsp(dist_matrix)
    # print(distance)

    return permutation


def assign_order(img_embs, assembly_id_dict, metadata_list, centers, route):
    # Compute line segments between two adjacent nodes on the route
    segments = dict()
    for i in range(len(route)):
        if i + 1 < len(route):
            segments[i] = {'start': centers[route[i]], 'end': centers[route[i + 1]]}

    segments[len(route) - 1] = {'start': centers[route[-1]], 'end': centers[route[0]]}

    emb_order_info_list = []
    for emb in img_embs:
        emb_segment_id, emb_projected_dist = compute_segment_id(emb, segments, route)
        emb_order_info_list.append({'seg_id_order': emb_segment_id, 'proj_dist': emb_projected_dist})

    ordered_assembly_parts = dict()
    for assembly_id, indexes in assembly_id_dict.items():
        # print(f"{assembly_id}: {indexes}")
        # print(emb_order_info_list[indexes[0]:indexes[-1]+1])
        emb_order_info_slice = emb_order_info_list[indexes[0]:indexes[-1]+1]
        sorted_parts = sorted(enumerate(emb_order_info_slice), key=lambda d: (d[1]['seg_id_order'], d[1]['proj_dist']))
        # print(sorted_parts)
        ordered_assembly_parts[assembly_id] = [metadata_list[indexes[i]] for i, _ in sorted_parts]

    return ordered_assembly_parts


def compute_segment_id(emb, segments, route):
    shortest_dist = 10000000
    closest_seg_id_order = 0
    nearest = np.zeros_like(emb)
    for seg_id, seg in segments.items():
    #     dist = norm(np.cross(seg['start'] - seg['end'], seg['end'] - emb)) / norm(seg['start'] - seg['end'])
        dist, proj_pnt = pnt2line(emb, seg['start'], seg['end'])
        if dist < shortest_dist:
            shortest_dist = dist
            closest_seg_id_order = route[seg_id]
            nearest = proj_pnt
    # projected_dist = math.sqrt(math.pow(math.dist(emb, segments[closest_seg_id]), 2) + math.pow(shortest_dist, 2))
    projected_dist = math.dist(emb, nearest)

    return closest_seg_id_order, projected_dist


def save_orders_to_file(orders, dirname, vid):
    print(f'Saving part orders to {dirname}/part_orders_{vid}.json')
    with open(dirname+'/'+f'part_orders_{vid}.json', 'w', encoding='utf8') as fp:
        json.dump(orders, fp, indent=4, ensure_ascii=False, sort_keys=False)

    return


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_path', type=str, help='Path to the file that saves part images')
    parser.add_argument('--metadata_path', type=str, help='Path to the file that saves embedding metadata')
    args = parser.parse_args()

    # Load in image embeddings
    img_embs = torch.load(args.embedding_path).detach().numpy()

    # Calculate embedding centers
    kmeans = KMeans(n_clusters=25, random_state=0, n_init="auto")
    centers = compute_center(img_embs, kmeans)

    # Calculate TSP route
    route = compute_route(centers)
    print(route)

    # Load assembly ids from the metadata
    with open(args.metadata_path, 'r') as f:
        metadata = json.load(f)

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

    orders = assign_order(img_embs, assembly_id_dict, metadata, centers, route)
    vid = args.embedding_path.split('/')[-1].split('.')[0].split('_')[-1]
    save_orders_to_file(orders, '../processed_data', vid)
