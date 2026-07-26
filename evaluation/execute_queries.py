# Test DB 
import os
from pathlib import Path
from pymongo import MongoClient
from pymongo.server_api import ServerApi


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


project_root = Path(__file__).resolve().parents[1]
env_file = project_root / "atlas-credentials.env"
data_dir = project_root / "data"
input_file = data_dir / "qwen_normalized.json"
output_file = data_dir / "qwen_execution_results.json"

env_values = load_env_file(env_file)
uri = env_values.get("MONGODB_URI") or os.getenv("MONGODB_URI")

if not uri:
    raise RuntimeError(f"MONGODB_URI not found in {env_file}")

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")

    db = client["concert_singer"]      # or pets_1, car_1, network_1
    print(db.list_collection_names())
    import json
    from pprint import pprint

    with input_file.open(encoding="utf-8") as f:
        questions = json.load(f)

    results = []

    for item in questions:
        query = item["normalized_query"]
        print("=" * 80)
        print(item["id"])

        try:
            result = eval(query)

            if isinstance(result, (int, float, str, bool)):
                pass
            elif not isinstance(result, list):
                result = list(result)

            results.append({
                "id": item["id"],
                "question": item["question"],
                "query": query,
                "status": "PASS",
                "result": result
            })

        except Exception as e:
            results.append({
                "id": item["id"],
                "question": item["question"],
                "query": query,
                "status": "FAIL",
                "error": str(e)
            })

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Finished")
except Exception as e:
    print(e)

