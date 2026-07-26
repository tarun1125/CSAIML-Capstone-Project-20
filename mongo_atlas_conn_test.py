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


env_file = Path(__file__).with_name("atlas-credentials.env")
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
    result = list(db.singer.find({}, {"_id": 0}))
    print(result[:5])
except Exception as e:
    print(e)

