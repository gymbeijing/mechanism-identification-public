import json

if __name__ == '__main__':
	with open('../processed_data/emb_idx_filtered.json') as fp:
		metadata = json.load(fp)

	assembly_id_list = []
	for md in metadata:
		assembly_id_list.append("_".join(md.split("_")[1:3]))

	#print(assembly_id_list[:10])
	assembly_id_set = set(assembly_id_list)
	assembly_id_list = list(assembly_id_set)

	with open(f'../processed_data/processed_assembly_ids.json', 'w', encoding='utf8') as fp:   # maybe filtered_assembly_ids.json is a better name
		json.dump(assembly_id_list, fp, indent=4, ensure_ascii=False, sort_keys=False)
	print(len(assembly_id_list))
