import ast
import json
import re
import sys

# Defaults keep this a no-argument, Qwen-only script like before. Pass two
# paths on the command line to reuse it for the GPT arm instead of
# duplicating the whole file:
#   python normalize.py data/gpt_results.json data/gpt_normalized.json
INPUT = sys.argv[1] if len(sys.argv) > 1 else "data/claude_baseline_results.json"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "data/claude_normalized.json"


def normalize(query: str) -> str:
    query = query.strip()

    try:
        tree = ast.parse(query, mode="eval")
    except SyntaxError:
        # Leave unparseable input alone. execute_queries.py will correctly
        # report this as a parse failure -- better than silently mangling it
        # into something that merely looks plausible.
        print(f"[normalize] WARNING could not parse, leaving as-is: {query[:80]}")
        return re.sub(r"\s+", " ", query).strip()

    node = tree.body
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "list"
        and len(node.args) == 1
    ):
        node = node.args[0]

    return ast.unparse(node)


if __name__ == "__main__":
    with open(INPUT, encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        item["normalized_query"] = normalize(item["generated_query"])
        print(f"[normalize] id={item.get('id')}: {item['normalized_query'][:80]}")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[normalize] wrote {len(data)} rows -> {OUTPUT}")
