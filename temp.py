import json


def load_json(filepath):
    with open(filepath, 'r') as fp:
        j = json.load(fp)  # .shape[0]=141245

    return j


if __name__ == '__main__':
    aid_result_map = load_json('./model_outputs/iou_list_by_assembly.json')
    total_iou_score = 0.0
    total = 0
    k = 1
    for aid, result_list in aid_result_map.items():
        for result in result_list:
            if result["num_nodes"] >= 10 and result["k"] == k:
                total_iou_score += sum(result["iou_score"])
                total += len(result["iou_score"])

    print(total_iou_score / total)

