import json
import numpy as np
from pathlib import Path

for filpath in Path(
    "/home/tom/projects/GMR/general_motion_retargeting/ik_configs"
).glob("*.json"):
    data = json.loads(filpath.read_text())

    ik_match_table = {}
    task_weights = []

    for key, val in data["ik_match_table1"].items():
        ik_match_table[key] = {
            "human_frame_name": val[0],
            "position": val[3],
            "rotation": val[4],
        }

    if data["use_ik_match_table1"]:
        task_weights.append(
            {
                key: {"position": val[1], "rotation": val[2]}
                for key, val in data["ik_match_table1"].items()
            }
        )

    if data["use_ik_match_table2"]:
        task_weights.append(
            {
                key: {"position": val[1], "rotation": val[2]}
                for key, val in data["ik_match_table2"].items()
            }
        )

    data.pop("ik_match_table1")
    data.pop("ik_match_table2")
    data.pop("use_ik_match_table1")
    data.pop("use_ik_match_table2")
    data["ik_match_table"] = ik_match_table
    data["task_weights"] = task_weights

    filpath.write_text(json.dumps(data, indent=4))
