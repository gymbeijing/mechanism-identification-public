"""
Build graphs for the Fusion Gallery assembly dataset
"""
import networkx as nx
import json
from collections import defaultdict
from tqdm import tqdm
import pickle
from joe_src.utils import *
    
def add_nodes(G, assembly_data, bodies):
    for b in assembly_data["bodies"]:
        if b in bodies:
            G.add_node(b)

def check_assembly(assembly_data):
    if not "contacts" in assembly_data:
        return False
    if assembly_data["contacts"] is None:
        return False
    return True


def add_edges(G, assembly_data, bodies):
    for c in assembly_data["contacts"]:
        b1 = c["entity_one"]["body"]
        b2 = c["entity_two"]["body"]
        if b1 in bodies and b2 in bodies:
            if b1 != b2:
                G.add_edge(b1, b2)

def make_graph(assembly_json, bodies):
    assembly_data = load_json(assembly_json)
    
    if not check_assembly(assembly_data):
        return None
    G = nx.Graph()
    add_nodes(G, assembly_data, bodies)
    add_edges(G, assembly_data, bodies)
    if G.number_of_edges() == 0:
        return None
    return G

def get_assembly_uuid(fullname):
    uuid = '_'.join(fullname.split('_')[:2])
    return uuid
    
def make_all_graphs(assembly_dataset, metadata_pathname):
    cache_file = assembly_dataset.parent / "graph_cache.p"
    if cache_file.exists():
        with open(cache_file, "rb") as fp:
            return pickle.load(fp)
        
    metadata = load_json(metadata_pathname)
    assemblies = assemblies_and_bodies(metadata)

    assembly_graphs = {}
    assembly_files = list(assembly_dataset.glob("*_assembly.json"))
    for assembly_file in tqdm(assembly_files):
        assembly_uuid = get_assembly_uuid(assembly_file.stem)
        if assembly_uuid in assemblies:
            bodies = assemblies[assembly_uuid]
            graph = make_graph(assembly_file, bodies)
            if graph is not None:
                assembly_graphs[assembly_uuid] = graph
            
    with open(cache_file, "wb") as fp:
        pickle.dump(assembly_graphs, fp)
        
    return assembly_graphs