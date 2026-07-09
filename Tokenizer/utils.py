import json


def save_json(data, filepath):

    with open(filepath, "w", encoding="utf8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)