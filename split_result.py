import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--result_file", type=str)
args = parser.parse_args()

with open(args.result_file, "r", encoding="utf-8") as f:
    result = json.load(f)

split_result = {"1": [], "2": [], "3": [], "4": [], "5": []}

for prediction in result:
    split_result[prediction["title"][-1]].append(
        {
            "title": prediction["title"][:-2],
            "h_idx": prediction["h_idx"],
            "t_idx": prediction["t_idx"],
            "r": prediction["r"]
        }
    )

for id in split_result:
    with open("result-{}.json".format(id), "w", encoding="utf-8") as f:
        json.dump(split_result[id], f)
