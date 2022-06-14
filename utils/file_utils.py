import re

import numpy as np

from utils.data_utils import downscale_tsp_coords

## <integer> <real> <real>
Regex_Node_Coord_Line = re.compile(r"(\d+)[\t ]([\d\.e\-\+]+)[\t ]([\d\.e\-\+]+)")


def load_tsplib_file(data_file, normalize=False):
    nodes_coord = []
    with open(data_file) as r:
        lines = r.read().replace("EOF", "").split("NODE_COORD_SECTION")[1].strip().split("\n")
        for line in lines:
            match = Regex_Node_Coord_Line.match(line)
            if not match:
                raise SystemExit(f"Error loading {data_file}: line '{line}'")
            nodes_coord.append([float(match.group(2)), float(match.group(3))])
    nodes_coord = np.array(nodes_coord)

    if normalize:
        nodes_coord = downscale_tsp_coords(nodes_coord)

    return nodes_coord
