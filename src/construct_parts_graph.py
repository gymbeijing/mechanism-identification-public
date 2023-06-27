import networkx as nx
import json
import logging
import argparse
import glob

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
			logging.info(f'contacts is null')
			return None
	else:
		logging.info(f'not having contacts in info.keys()')
		return None


def get_graphs(assembly_ids, assembly_folder):
	graph_dict = dict()
	for a_id in assembly_ids:
		#print(a_id)
		a_info = read_json(f'{assembly_folder}/{a_id}_assembly.json')
		a_graph = build_graph(a_info)
		#print(a_graph.nodes)
		#print(a_graph.edges)
		if a_graph is not None:
			graph_dict[a_id] = a_graph
		else:
			logging.info(f'{a_id}')
	
	return graph_dict
	


if __name__ == '__main__':
	# Parse the arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--assembly_id_path', type=str, help='Path to the file that stores all the processed assembly ids')
	parser.add_argument('--assembly_folder', type=str, help='Path to the folder that stores all the assembly.json files')
	args = parser.parse_args()

	# Load in the assembly ids from the view folders into a list
	assembly_id_list = read_json(args.assembly_id_path)   # should be ../processed_data/processed_assembly_ids.json
	logging.info(f'Found {len(assembly_id_list)} assembly ids in {args.assembly_id_path}')

	# Load in the paths of assembly json files into a list
	# Maybe not necessary
	assembly_folder = args.assembly_folder   # should be ../raw_data/assembly
	assembly_json_list = glob.glob(f'{args.assembly_folder}/*.json')
	logging.info(f'Found {len(assembly_json_list)} assembly json files in {args.assembly_folder}')

	#print(assembly_json_list[:10])
	assembly_graph_dict = get_graphs(assembly_id_list, args.assembly_folder)
	print(len(assembly_graph_dict.keys()))
