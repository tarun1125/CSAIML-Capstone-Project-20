import json
import re

INPUT = "data/qwen2.5-coder_results.json"
OUTPUT = "data/qwen_normalized.json"

def normalize(query: str) -> str:
    query = query.strip()

    query = re.sub(r"^list\((.*)\)$", r"\1", query)

    query = query.replace("'", '"')

    query = re.sub(r"\s+", " ", query)

    return query.strip()


with open(INPUT, encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    item["normalized_query"] = normalize(item["generated_query"])

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("Done")