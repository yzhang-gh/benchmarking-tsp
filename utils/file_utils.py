import numpy as np
import re

## <integer> <real> <real>
Regex_Node_Coord_Line = re.compile(r"([\d]+) ([\d\.e\-\+]+) ([\d\.e\-\+]+)")


def load_tsplib_file(data_file):
    nodes_coord = []
    with open(data_file) as r:
        lines = r.read().replace("EOF", "").split("NODE_COORD_SECTION")[1].strip().split("\n")
        for line in lines:
            match = Regex_Node_Coord_Line.match(line)
            if not match:
                print(f"'{line}' not match")
                raise SystemExit("Error loading {data_file}: line '{line}'")
            nodes_coord.append([float(match.group(2)), float(match.group(3))])
    nodes_coord = np.array(nodes_coord)
    return nodes_coord
