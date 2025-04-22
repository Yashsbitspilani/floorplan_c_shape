# """
# fetch_data.py

# Defines a C‑shaped boundary, room dimensions, and adjacency graph
# for the floor‑plan placement problem.
# """

# import numpy as np
# from shapely.geometry import Polygon

# def processed_data():
#     """
#     Returns:
#       - grid_size: 2‑tuple (sx, sy) giving the bounding box of the plot.
#       - boundary: a Shapely Polygon representing the non‑rectangular plot.
#       - tiles: dict mapping tile_id (0..n-1) to (width, height).
#       - graph: adjacency matrix of shape (n, n), with 1 = desired adjacency.
#       - rotate: boolean, whether we allow 90° rotations of rooms.
#     """
#     # 1) Grid size (for initial layout and bounds checking)
#     sx, sy = 12, 10

#     # 2) C‑shaped boundary: outer rectangle minus an inner cut‑out
#     outer = Polygon([(0,0), (sx,0), (sx,sy), (0,sy)])
#     inner = Polygon([(6, 3), (sx, 3), (sx, 7), (6, 7)])
#     boundary = outer.difference(inner)

#     # 3) Room definitions (up to 10 rooms). Each is (width, height).
#     #    Adapt these to your assignment’s room dimensions.
#     tiles = {
#         0: (3, 5),
#         1: (6, 4),
#         2: (5, 5),
#         3: (4, 4),
#         4: (5, 6),
#         5: (4, 3),
#         6: (5, 5),
#         7: (7, 3),
#         8: (5, 7),
#         9: (4, 3),
#     }

#     # 4) Adjacency graph: N×N numpy array where graph[i,j]=1 iff rooms i and j should touch.
#     #    Adapt this to your assignment’s adjacency list.
#     graph = np.zeros((len(tiles), len(tiles)), dtype=int)
#     edges = [
#         (0,1),(0,3),(1,2),(2,4),(3,5),
#         (4,5),(4,6),(6,7),(7,8),(8,9)
#     ]
#     for u, v in edges:
#         graph[u, v] = graph[v, u] = 1

#     # 5) Allow rotations of rooms?
#     rotate = True

#     return (sx, sy), boundary, tiles, graph, rotate

# """
# fetch_data.py

# Defines a true C‑shaped boundary, room dimensions, and adjacency graph
# for the floor‑plan placement problem.
# (Here we’ve reduced each room’s dimensions so they fit comfortably.)
# """

# import numpy as np
# from shapely.geometry import Polygon

# def processed_data():
#     """
#     Returns:
#       - grid_size: (sx, sy) bounding box of the plot (12×10).
#       - boundary: a Shapely Polygon for a right‑facing 'C' shape.
#       - tiles: dict mapping room_id (0..9) → (width, height).
#       - graph: adjacency matrix (10×10) with 1 = desired adjacency.
#       - rotate: bool, whether 90° rotations are allowed.
#     """
#     # 1) Grid size
#     sx, sy = 12, 10

#     # 2) C‑shaped boundary: remove a right‑middle rectangle
#     outer = Polygon([(0, 0), (sx, 0), (sx, sy), (0, sy)])
#     inner = Polygon([(6, 3), (sx, 3), (sx, 7), (6, 7)])  # cut‑out on the right
#     boundary = outer.difference(inner)

#     # 3) Room sizes (all smaller now!)
#     tiles = {
#         0: (2, 2),
#         1: (3, 1),
#         2: (2, 1),
#         3: (1, 2),
#         4: (2, 2),
#         5: (1, 1),
#         6: (2, 2),
#         7: (3, 2),
#         8: (2, 3),
#         9: (1, 1),
#     }
    

#     # 4) Adjacency graph (symmetric 10×10)
#     graph = np.zeros((10, 10), dtype=int)
#     edges = [
#         (0,1), (0,3), (1,2), (2,4), (3,5),
#         (4,5), (4,6), (6,7), (7,8), (8,9)
#     ]
#     for u, v in edges:
#         graph[u, v] = graph[v, u] = 1

#     # 5) Allow rotations?
#     rotate = True

#     return (sx, sy), boundary, tiles, graph, rotate


# """
# fetch_data.py

# Loads level‑data from JSON (one file per level), and returns:
#   - grid_size: (rows, cols)
#   - boundary: a Shapely Polygon of the C-shaped plot
#   - tiles: dict tid→numpy array of 0/1 shape
#   - graph: NxN numpy adjacency matrix
#   - rotate: bool (allow rotations)
# """

# import json
# import numpy as np
# from shapely.geometry import Polygon

# def processed_data(level):
#     # 1) Load the JSON file for this level
#     path = f"levels/level_{level}.json"
#     with open(path) as f:
#         data = json.load(f)

#     # 2) Grid dims
#     rows = data["grid"]["row_size"]
#     cols = data["grid"]["column_size"]

#     # 3) Build a C-shaped plot boundary (outer minus inner on the right)
#     outer = Polygon([(0,0), (cols,0), (cols,rows), (0,rows)])
#     # cut out a rectangle on the right half, vertically centered
#     cut_left = cols * 0.5
#     cut_bottom = rows * 0.25
#     cut_top = rows * 0.75
#     inner = Polygon([
#         (cut_left,    cut_bottom),
#         (cols,        cut_bottom),
#         (cols,        cut_top),
#         (cut_left,    cut_top),
#     ])
#     boundary = outer.difference(inner)

#     # 4) Parse each node’s shape into a numpy array
#     tiles = {}
#     for node in data["graph"]["nodes"]:
#         tid = node["id"] - 1                # convert 1-based→0-based
#         shape = np.array(node["shape"], int)
#         tiles[tid] = shape

#     # 5) Build adjacency matrix
#     n = len(tiles)
#     graph = np.zeros((n,n), int)
#     for u,v in data["graph"]["edges"]:
#         graph[u][v] = graph[v][u] = 1

#     # 6) Allow 90° rotations
#     rotate = True

#     return (rows, cols), boundary, tiles, graph, rotate

# fetch_data.py

# import os
# import json
# import numpy as np
# from shapely.geometry import Polygon

# def processed_data(level):
#     """
#     Loads level_N.json from the 'levels' folder beside this file.
#     Returns (grid_size, boundary, tiles, graph, rotate).
#     """
#     # 1) Compute absolute path to the JSON
#     base = os.path.dirname(os.path.abspath(__file__))
#     path = os.path.join(base, "levels", f"level_{level}.json")

#     # 2) Fail fast if missing
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Level file not found: {path}")

#     # 3) Load JSON
#     with open(path, 'r') as f:
#         data = json.load(f)

#     # 4) Extract grid size
#     rows = data["grid"]["row_size"]
#     cols = data["grid"]["column_size"]

#     # 5) Build C-shaped boundary (outer minus inner)
#     outer = Polygon([(0,0), (cols,0), (cols,rows), (0,rows)])
#     # cut-out on the right
#     cut_left = cols * 0.5
#     cut_bottom = rows * 0.25
#     cut_top = rows * 0.75
#     inner = Polygon([
#         (cut_left, cut_bottom),
#         (cols,     cut_bottom),
#         (cols,     cut_top),
#         (cut_left, cut_top),
#     ])
#     boundary = outer.difference(inner)

#     # 6) Parse tiles: id→numpy shape
#     tiles = {}
#     for node in data["graph"]["nodes"]:
#         tid = node["id"] - 1  # convert 1-based to 0-based
#         tiles[tid] = np.array(node["shape"], dtype=int)

#     # 7) Adjacency matrix
#     n = len(tiles)
#     graph = np.zeros((n,n), dtype=int)
#     for u,v in data["graph"]["edges"]:
#         graph[u][v] = graph[v][u] = 1

#     return (rows, cols), boundary, tiles, graph, True  # rotate=True

# """
# fetch_data.py

# Defines:
#   - A true C‑shaped boundary via Shapely
#   - Up to 10 room‐shape arrays (numpy 0/1)
#   - An adjacency graph (numpy matrix)
#   - A rotate flag
# Returns all of the above in processed_data().
# """

# import numpy as np
# from shapely.geometry import Polygon

# def processed_data():
#     # 1) Plot size (cols × rows)
#     cols, rows = 12, 10

#     # 2) C‑shaped boundary: outer minus inner cut‑out on right
#     outer = Polygon([(0, 0), (cols, 0), (cols, rows), (0, rows)])
#     inner = Polygon([(6, 3), (cols, 3), (cols, 7), (6, 7)])
#     boundary = outer.difference(inner)

#     # 3) Room shapes as small matrices: 1 = occupied cell
#     #    (Here 7 rooms are shown; add up to 10 as needed)
#     tiles = {
#         0: np.array([[1,1],[1,1]]),       # 2×2 block
#         1: np.array([[1,1,1]]),           # 1×3 block
#         2: np.array([[1],[1],[1]]),       # 3×1 block
#         3: np.array([[1,1,1,1]]),         # 1×4 block
#         4: np.array([[1,1],[1,0]]),       # L‑shape
#         5: np.array([[1]]),               # 1×1 cell
#         6: np.array([[1,1,1],[1,0,0]]),   # ┏‑shape
#         7: np.array([[1,1],[1,1],[1,1]]), # 3×2 block
#         8: np.array([[1,0],[1,1],[1,0]]), # T‑shape
#         9: np.array([[1,1,1]]),           # another 1×3
#     }

#     # 4) Adjacency: list of pairs that *should* touch
#     edges = [
#         (0,1),(0,2),(1,4),(2,5),
#         (4,5),(4,6),(6,7),(7,8),
#         (8,9),(3,9)    # etc.
#     ]
#     n = len(tiles)
#     graph = np.zeros((n,n), dtype=int)
#     for u, v in edges:
#         graph[u, v] = graph[v, u] = 1

#     # 5) Allow 90° rotations
#     rotate = True

#     # Return: (grid_size, boundary, tiles, graph, rotate)
#     return (rows, cols), boundary, tiles, graph, rotate

# """
# fetch_data.py

# Defines a true C‑shaped boundary, up to 10 room shapes, and the adjacency graph.
# Returns (grid_size, boundary, tiles, graph, rotate).
# """

# import numpy as np
# from shapely.geometry import Polygon

# def processed_data():
#     # 1) Grid dims: rows × cols
#     rows, cols = 10, 12

#     # 2) C‑shaped boundary: outer minus a right‑middle cut‑out
#     outer = Polygon([(0,0), (cols,0), (cols,rows), (0,rows)])
#     inner = Polygon([(6,3), (cols,3), (cols,7), (6,7)])
#     boundary = outer.difference(inner)

#     # 3) Room shapes as 0/1 numpy arrays (1 = cell occupied).
#     tiles = {
#         0: np.array([[1,1],[1,1]]),       # 2×2 block
#         1: np.array([[1,1,1]]),           # 1×3 block
#         2: np.array([[1],[1],[1]]),       # 3×1 block
#         3: np.array([[1,1,1,1]]),         # 1×4 block
#         4: np.array([[1,1],[1,0]]),       # L‑shape
#         5: np.array([[1]]),               # 1×1 cell
#         6: np.array([[1,1,1],[1,0,0]]),   # ┏‑shape
#         7: np.array([[1,1],[1,1],[1,1]]), # 3×2 block
#         8: np.array([[1,0],[1,1],[1,0]]), # T‑shape
#         9: np.array([[1,1,1]]),           # another 1×3
#     }

#     # 4) Adjacency edges: list of pairs (must touch exactly these)
#     edges = [
#         (0,1),(0,2),(1,4),(2,5),
#         (4,5),(4,6),(6,7),(7,8),
#         (8,9),(3,9)
#     ]
#     n = len(tiles)
#     graph = np.zeros((n,n), dtype=int)
#     for u,v in edges:
#         graph[u,v] = graph[v,u] = 1

#     # 5) Allow 90° rotations
#     rotate = True

#     return (rows, cols), boundary, tiles, graph, rotate

# import numpy as np
# from shapely.geometry import Polygon

# def processed_data():
#     # 1) Grid dims
#     rows, cols = 10, 12

#     # 2) C‑shaped boundary: outer minus inner cut‑out
#     outer = Polygon([(0,0),(cols,0),(cols,rows),(0,rows)])
#     inner = Polygon([(6,3),(cols,3),(cols,7),(6,7)])  # right‑middle hole
#     boundary = outer.difference(inner)               # 

#     # 3) Room shapes: dict id→0/1 numpy arrays
#     tiles = {
#         0: np.array([[1,1],[1,1]]),   # 2×2
#         1: np.array([[1,1,1]]),       # 1×3
#         # … up to 10 rooms …
#     }

#     # 4) Adjacency graph
#     edges = [(0,1),(0,2),(1,4),(2,5),(4,5),(4,6),(6,7),(7,8),(8,9),(3,9)]
#     n = len(tiles)
#     graph = np.zeros((n,n), int)
#     for u,v in edges:
#         graph[u,v] = graph[v,u] = 1          # 

#     return (rows, cols), boundary, tiles, graph, True  # rotate=True

# import numpy as np
# from shapely.geometry import Polygon

# def processed_data():
#     """
#     Returns:
#       - (rows, cols): grid dimensions
#       - boundary: C‑shaped Shapely Polygon
#       - tiles: dict room_id → 0/1 numpy array
#       - graph: N×N adjacency matrix
#       - rotate: bool
#     """
#     # 1) Grid size
#     rows, cols = 10, 12

#     # 2) C‑shaped boundary
#     outer = Polygon([(0, 0), (cols, 0), (cols, rows), (0, rows)])
#     inner = Polygon([(6, 3), (cols, 3), (cols, 7), (6, 7)])
#     boundary = outer.difference(inner)

#     # 3) Room shapes
#     tiles = {
#         0: np.array([[1,1],[1,1]]),
#         1: np.array([[1,1,1]]),
#         2: np.array([[1],[1],[1]]),
#         3: np.array([[1,1,1,1]]),
#         4: np.array([[1,1],[1,0]]),
#         5: np.array([[1]]),
#         6: np.array([[1,1,1],[1,0,0]]),
#         7: np.array([[1,1],[1,1],[1,1]]),
#         8: np.array([[1,0],[1,1],[1,0]]),
#         9: np.array([[1,1,1]]),
#     }

#     # 4) Desired adjacency edges
#     edges = [
#         (0,1),(0,2),(1,4),(2,5),
#         (4,5),(4,6),(6,7),(7,8),
#         (8,9),(3,9)
#     ]

#     # 5) Derive n from max tile ID, ensure edges fit
#     max_id = max(tiles.keys())
#     n = max_id + 1

#     # Assert no edge references outside 0..n-1
#     for u, v in edges:
#         assert 0 <= u < n and 0 <= v < n, f"Edge {(u,v)} out of bounds for {n} tiles"

#     # 6) Build adjacency matrix
#     graph = np.zeros((n, n), dtype=int)
#     for u, v in edges:
#         graph[u, v] = graph[v, u] = 1

#     # 7) Allow rotations
#     rotate = True

#     return (rows, cols), boundary, tiles, graph, rotate

import numpy as np
from shapely.geometry import Polygon

def processed_data():
    """
    Returns:
      - (rows, cols): grid dimensions
      - boundary: C-shaped Shapely Polygon
      - tiles: dict room_id → 0/1 numpy array
      - graph: N×N adjacency matrix (0/1)
      - rotate: bool
    """
    # 1) Grid size
    rows, cols = 10, 12

    # 2) C-shaped boundary
    outer = Polygon([(0, 0), (cols, 0), (cols, rows), (0, rows)])
    inner = Polygon([(6, 3), (cols, 3), (cols, 7), (6, 7)])
    boundary = outer.difference(inner)

    # 3) Room shapes (0/1 arrays)
    tiles = {
        0: np.array([[1,1],[1,1]]),
        1: np.array([[1,1,1]]),
        2: np.array([[1],[1],[1]]),
        3: np.array([[1,1,1,1]]),
        4: np.array([[1,1],[1,0]]),
        5: np.array([[1]]),
        6: np.array([[1,1,1],[1,0,0]]),
        7: np.array([[1,1],[1,1],[1,1]]),
        8: np.array([[1,0],[1,1],[1,0]]),
        9: np.array([[1,1,1]]),
    }

    # 4) Desired adjacency edges
    edges = [
        (0,1),(0,2),(1,4),(2,5),
        (4,5),(4,6),(6,7),(7,8),
        (8,9),(3,9)
    ]

    # 5) Build adjacency matrix
    n = max(tiles.keys()) + 1
    graph = np.zeros((n, n), dtype=int)
    for u, v in edges:
        graph[u, v] = graph[v, u] = 1

    # 6) Allow rotations
    rotate = True

    return (rows, cols), boundary, tiles, graph, rotate
