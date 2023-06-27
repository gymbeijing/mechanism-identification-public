import networkx as nx
import json
import logging
import argparse
import glob
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


def read_json(file_path):
	"""
	:param file_path: relative file path of the json file
	:return: a dict / list
	"""
	with open(file_path, 'r') as f:
		content = json.load(f)
	return content
	

def build_graph(info):

	G = nx.Graph()
	bodies = info["bodies"]
	G.add_nodes_from(bodies)

	if "contacts" in info.keys():
		contacts = info["contacts"]
		if contacts is not None:
			for contact in contacts:
				entity_one = contact["entity_one"]
				entity_two = contact["entity_two"]
				body_one = entity_one["body"]
				body_two = entity_two["body"]
				G.add_edge(body_one, body_two)

			return G
		else:
			#logging.info(f'contacts is null')
			return None
	else:
		#logging.info(f'not having contacts in info.keys()')
		return None


def compute_graphs(assembly_ids, assembly_folder):

	graph_dict = dict()
	for a_id in assembly_ids:
		#print(a_id)
		a_info = read_json(f'{assembly_folder}/{a_id}_assembly.json')
		a_graph = build_graph(a_info)
		#print(a_graph.nodes)
		#print(a_graph.edges)
		if a_graph is not None:
			graph_dict[a_id] = a_graph
		#else:
		#	logging.info(f'{a_id}')
	
	return graph_dict


def order_parts(neighbours, assembly_parts_to_order_map):

	neighbours_to_order_map = {part: assembly_parts_to_order_map[part] for part in neighbours if part in assembly_parts_to_order_map}
	sorted_neighbours_order_tuple_list = sorted(neighbours_to_order_map.items(), key=lambda x: x[1]) 
	ordered_neighbours = [item[0] for item in sorted_neighbours_order_tuple_list]

	return ordered_neighbours


def compute_part_graphs(graph_dict, all_parts_order):
	
	part_graph_dict = dict()
	for a_id, a_graph in tqdm(graph_dict.items()):
		a_nodes = a_graph.nodes
		part_graph_dict[a_id] = dict()   # warning: didn't perform further check, might include assembly having empty part graph
		a_parts_order = all_parts_order[a_id]   # list

		a_parts_to_order_map = dict()
		for idx, part in enumerate(a_parts_order):
			a_parts_to_order_map[part] = idx

		for a_node in a_nodes:
			#print(a_node)
			a_neighbours = a_graph.neighbors(a_node)
			a_neighbours = [f'0000_{a_id}_{a_n}_0000'for a_n in a_neighbours]
			if len(a_neighbours) != 0:
				ordered_a_neighbours = order_parts(a_neighbours, a_parts_to_order_map)
				if len(ordered_a_neighbours) != 0:
					part_graph_dict[a_id][f'0000_{a_id}_{a_node}_0000'] = ordered_a_neighbours
	
	return part_graph_dict


def write_json(content, filepath):

	with open(filepath, 'w', encoding='utf8') as fp:
		json.dump(content, fp, indent=4, ensure_ascii=False, sort_keys=False)
	
	return


if __name__ == '__main__':
	# Parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--assembly_id_path', type=str, help='Path to the file that stores all the processed assembly ids')
	parser.add_argument('--assembly_folder', type=str, help='Path to the folder that stores all the assembly.json files')
	parser.add_argument('--assembly_parts_order', type=str, help='Path to the file that stores the mapping from assembly id and computed orders of its parts')
	parser.add_argument('--save_path', type=str, help='Path to the file that will save the part graphs')
	args = parser.parse_args()

	# Load in the assembly ids from the view folders into a list
	assembly_id_list = read_json(args.assembly_id_path)   # should be ../processed_data/processed_assembly_ids.json
	logging.info(f'Found {len(assembly_id_list)} assembly ids in {args.assembly_id_path}')

	# Load in the paths of assembly json files into a list
	# Maybe not necessary
	assembly_folder = args.assembly_folder   # should be ../raw_data/assembly
	assembly_json_list = glob.glob(f'{args.assembly_folder}/*.json')
	logging.info(f'Found {len(assembly_json_list)} assembly json files in {args.assembly_folder}')

	# Compute whole graph for each assembly
	assembly_graph_dict = compute_graphs(assembly_id_list, args.assembly_folder)
	logging.info(f'Found {len(assembly_graph_dict.keys())} assembly json files that have proper contacts information')

	assembly_parts_order = read_json(args.assembly_parts_order)
	logging.info(f'Computing part graphs')
	part_graph_dict = compute_part_graphs(assembly_graph_dict, assembly_parts_order)

	logging.info(f'Saving part graphs into {args.save_path}')
	write_json(part_graph_dict, args.save_path)

	total = 0
	for p_graphs in part_graph_dict.values():
		total += len(p_graphs.keys())

	logging.info(f'Found {total} part graphs.')

