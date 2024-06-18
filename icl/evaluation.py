import argparse
import json
import os
import re


def match(predicted_relation, rel2id):
    if predicted_relation in rel2id:
        return True, rel2id[predicted_relation]
    return False, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file_path", type=str, default=r"datasets/re-docred/test_revised.json")
    parser.add_argument("--relation_file_path", type=str, default=r"datasets/re-docred/rel_info.json")
    parser.add_argument("--predictions_dir", type=str,
                        default="icl/vanilla_predictions/gpt-3.5-turbo-0125-1-shot/re-docred")
    args = parser.parse_args()

    with open(args.test_file_path, "r", encoding="utf-8") as f:
        documents = json.load(f)
    with open(args.relation_file_path, "r", encoding="utf-8") as f:
        relations = json.load(f)

    rel2id = {}
    for rel_id, rel in relations.items():
        rel2id[rel] = rel_id

    total_labels = 0
    total_predictions = 0
    total_matches = 0
    predictions_file_list = os.listdir(args.predictions_dir)
    for predictions_file in predictions_file_list:
        labels = documents[int(predictions_file)]["labels"]
        labels_set = set()
        for label in labels:
            labels_set.add((label["h"], label["r"], label["t"]))
        total_labels += len(labels_set)

        predictions_set = set()
        predictions_file_path = os.path.join(args.predictions_dir, predictions_file)
        with open(predictions_file_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                prediction = line.strip()
                m = re.match(r"<\[(\d+)\];\s+([a-z\s,]+);\s+\[(\d+)\]>", prediction)
                if m:
                    match_flag, match_relation = match(m.group(2).strip(), rel2id)
                    h, t = int(m.group(1)) - 1, int(m.group(3)) - 1
                    if match_flag and h != t:
                        predictions_set.add((h, match_relation, t))
        total_predictions += len(predictions_set)

        total_matches += len(labels_set & predictions_set)

    precision = 100 * total_matches / total_predictions
    recall = 100 * total_matches / total_labels
    f1 = 2 * precision * recall / (precision + recall)
    print("Precision: {:.2f}%".format(precision))
    print("Recall: {:.2f}%".format(recall))
    print("F1: {:.2f}%".format(f1))


if __name__ == "__main__":
    main()
