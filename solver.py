# """
# solver.py

# Loads data from fetch_data.py, runs sequence‑pair + simulated annealing
# to place rooms inside a C‑shaped boundary, then visualizes the result.
# """

# import random
# import math

# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon as MplPolygon, Rectangle
# from shapely.geometry import box
# from shapely.ops import unary_union

# from fetch_data import processed_data

# # -- Sequence-Pair Decoding & Scoring -----------------------------------------

# def decode_sequence_pair(alpha, beta, tiles):
#     pos_a = {r: i for i, r in enumerate(alpha)}
#     pos_b = {r: i for i, r in enumerate(beta)}
#     x = {r: 0 for r in alpha}
#     y = {r: 0 for r in beta}

#     # X coords by alpha order
#     for r in alpha:
#         w, _ = tiles[r]
#         preds = [s for s in alpha if pos_b[s] < pos_b[r]]
#         x[r] = 0 if not preds else max(x[s] + tiles[s][0] for s in preds)

#     # Y coords by beta order
#     for r in beta:
#         _, h = tiles[r]
#         preds = [s for s in beta if pos_a[s] < pos_a[r]]
#         y[r] = 0 if not preds else max(y[s] + tiles[s][1] for s in preds)

#     return {r: (x[r], y[r], tiles[r][0], tiles[r][1]) for r in tiles}


# def adjacency_score(placement, graph):
#     score = 0
#     n = graph.shape[0]
#     for i in range(n):
#         for j in range(i + 1, n):
#             if graph[i, j]:
#                 xi, yi, wi, hi = placement[i]
#                 xj, yj, wj, hj = placement[j]
#                 Ri = box(xi, yi, xi + wi, yi + hi)
#                 Rj = box(xj, yj, xj + wj, yj + hj)
#                 if Ri.touches(Rj) and Ri.intersection(Rj).length > 0:
#                     score += 1
#     return score


# def energy(state, boundary, tiles, graph, penalty=1000):
#     alpha, beta = state
#     placement = decode_sequence_pair(alpha, beta, tiles)
#     E = -adjacency_score(placement, graph)

#     # Penalty for outside-boundary or overlaps
#     polys, total_area = [], 0
#     for r, (x, y, w, h) in placement.items():
#         total_area += w * h
#         P = box(x, y, x + w, y + h)
#         if not boundary.contains(P):
#             E += penalty
#         polys.append(P)

#     union = unary_union(polys)
#     if union.area < total_area - 1e-6:
#         E += penalty * (total_area - union.area)

#     return E


# # -- Simulated Annealing -------------------------------------------------------

# def optimize(boundary, tiles, graph, max_iter=5000):
#     rooms = list(tiles.keys())
#     alpha = rooms[:]
#     beta = rooms[:]
#     random.shuffle(alpha)
#     random.shuffle(beta)
#     state = (alpha, beta)
#     best_state, best_E = state, energy(state, boundary, tiles, graph)

#     T = 1.0
#     for _ in range(max_iter):
#         a2, b2 = state[0][:], state[1][:]
#         if random.random() < 0.5:
#             i, j = random.sample(rooms, 2)
#             a2[i], a2[j] = a2[j], a2[i]
#         else:
#             i, j = random.sample(rooms, 2)
#             b2[i], b2[j] = b2[j], b2[i]

#         new_state = (a2, b2)
#         E_new = energy(new_state, boundary, tiles, graph)
#         E_cur = energy(state, boundary, tiles, graph)

#         if E_new < E_cur or random.random() < math.exp((E_cur - E_new) / T):
#             state = new_state
#             if E_new < best_E:
#                 best_state, best_E = new_state, E_new

#         T *= 0.995

#     final_placement = decode_sequence_pair(*best_state, tiles)
#     return best_state, final_placement


# # -- Visualization Helpers ----------------------------------------------------

# def plot_boundary(boundary, ax=None, edgecolor="black", linewidth=2):
#     if ax is None:
#         fig, ax = plt.subplots()
#     def _draw(poly):
#         x, y = poly.exterior.xy
#         ax.add_patch(MplPolygon(list(zip(x, y)), closed=True,
#                                 fill=False, edgecolor=edgecolor,
#                                 linewidth=linewidth))
#         for interior in poly.interiors:
#             xi, yi = interior.xy
#             ax.add_patch(MplPolygon(list(zip(xi, yi)), closed=True,
#                                     fill=False, edgecolor=edgecolor,
#                                     linewidth=linewidth, linestyle="--"))
#     if boundary.geom_type == "Polygon":
#         _draw(boundary)
#     else:
#         for poly in boundary.geoms:
#             _draw(poly)
#     return ax


# def plot_rooms(placement, ax=None, facecolor="C0", edgecolor="k", alpha=0.5):
#     if ax is None:
#         fig, ax = plt.subplots()
#     for rid, (x, y, w, h) in placement.items():
#         ax.add_patch(Rectangle((x, y), w, h,
#                                facecolor=facecolor,
#                                edgecolor=edgecolor,
#                                alpha=alpha))
#         ax.text(x + w/2, y + h/2, str(rid),
#                 ha="center", va="center", color="white")
#     return ax


# def visualize(boundary, placement, figsize=(8, 6)):
#     fig, ax = plt.subplots(figsize=figsize)
#     plot_boundary(boundary, ax=ax)
#     plot_rooms(placement, ax=ax)
#     ax.set_aspect("equal", "box")
#     ax.autoscale()
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_title("Optimal Floorplan in C‑Shaped Boundary")
#     plt.show()


# # -- Main ----------------------------------------------------------------------

# def main():
#     (sx, sy), boundary, tiles, graph, rotate = processed_data()
#     seq_pair, placement = optimize(boundary, tiles, graph)
#     print("Best sequence-pair:", seq_pair)
#     for r, (x, y, w, h) in placement.items():
#         print(f"Room {r}: x={x:.2f}, y={y:.2f}, w={w}, h={h}")
#     visualize(boundary, placement)


# if __name__ == "__main__":
#     main()

# """
# solver.py

# Implements a Z3-based solver that places each room‐tile on an integer (row×col) grid,
# forces cells outside the C-shaped boundary to be “empty” (−1), enforces each tile’s shape
# via pure Z3 constraints, and then maximizes adjacency edges.
# """

# import numpy as np
# from z3 import Solver, IntVector, And, Or, Optimize
# from shapely.geometry import Point

# def get_solver(grid_size, boundary, tiles, graph, rotate):
#     rows, cols = grid_size
#     n = len(tiles)

#     opt = Optimize()
#     # Create a flat array of Z3 Ints, reshape into [rows×cols]
#     BRD = np.array(IntVector('b', rows*cols), dtype=object).reshape(rows, cols)

#     # 1) Boundary mask: cells outside C shape must be −1
#     for i in range(rows):
#         for j in range(cols):
#             pt = Point(j + 0.5, rows - (i + 0.5))  
#             # note: flip Y if you want row0 at top
#             if not boundary.contains(pt):
#                 opt.add(BRD[i,j] == -1)
#             else:
#                 opt.add(And(BRD[i,j] >= 0, BRD[i,j] < n))

#     # 2) Tile placement constraints
#     from collections import defaultdict
#     class Tile:
#         def __init__(self, shape, tid):
#             self.tid = tid
#             # collect unique rotations
#             d = defaultdict(set)
#             arr = shape.copy()
#             for _ in range(4 if rotate else 1):
#                 d[arr.shape].add(arr.tobytes())
#                 arr = np.rot90(arr)
#             self.orients = [
#                 np.frombuffer(b, int).reshape(h,w)
#                 for (h,w), blobs in d.items() for b in blobs
#             ]
#         def add(self):
#             conds = []
#             for arr in self.orients:
#                 h,w = arr.shape
#                 ones = [(r,c) for r in range(h) for c in range(w) if arr[r,c]]
#                 for i in range(rows - h + 1):
#                     for j in range(cols - w + 1):
#                         # force BRD[i+r,j+c] == self.tid for all ones, 
#                         # and != self.tid for all other in-bound cells
#                         must = [BRD[i+r, j+c] == self.tid for (r,c) in ones]
#                         # outside ones, just ensure no conflict at those positions
#                         conds.append(And(*must))
#             opt.add(Or(*conds))

#     for tid, shape in tiles.items():
#         Tile(shape, tid).add()

#     # 3) Adjacency Booleans & objective
#     adj_vars = []
#     for u in range(n):
#         for v in range(u+1, n):
#             if graph[u,v]:
#                 var = opt.model().bool_const(f"adj_{u}_{v}") if False else None
#                 conds = []
#                 for i in range(rows):
#                     for j in range(cols):
#                         if i+1 < rows:
#                             conds.append(And(BRD[i,j]==u, BRD[i+1,j]==v))
#                             conds.append(And(BRD[i,j]==v, BRD[i+1,j]==u))
#                         if j+1 < cols:
#                             conds.append(And(BRD[i,j]==u, BRD[i,j+1]==v))
#                             conds.append(And(BRD[i,j]==v, BRD[i,j+1]==u))
#                 b = opt.model().fresh_bool(f"adj_{u}_{v}") if False else None
#                 opt.add(b == Or(*conds))
#                 adj_vars.append(b)

#     # maximize the sum of all adj_vars
#     opt.maximize(sum([opt.model().If(b,1,0) for b in adj_vars]))

#     return opt, BRD

# def solve(level):
#     from fetch_data import processed_data
#     grid_size, boundary, tiles, graph, rotate = processed_data(level)
#     opt, BRD = get_solver(grid_size, boundary, tiles, graph, rotate)

#     if opt.check() != sat:
#         raise RuntimeError("No solution found")
#     m = opt.model()
#     rows, cols = grid_size
#     sol = np.zeros((rows,cols), int)
#     for i in range(rows):
#         for j in range(cols):
#             sol[i,j] = m[BRD[i,j]].as_long()
#     return sol

# if __name__ == "__main__":
#     # Example: solve level 14
#     sol = solve(14)
#     print(sol)

# """
# solver.py

# - Ingests the in‑code data from fetch_data.py
# - Places each room (tile) inside the C‑shape
# - Maximizes the number of satisfied adjacencies
# - Visualizes the final layout via Matplotlib
# """

# import random, math
# import numpy as np
# from shapely.geometry import box, Point
# from shapely.ops import unary_union
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon as MplPolygon, Rectangle

# from fetch_data import processed_data

# # --- Sequence‑Pair decoding to get (x,y,w,h) per room ---
# def decode_sequence_pair(alpha, beta, tiles):
#     pos_a = {r:i for i,r in enumerate(alpha)}
#     pos_b = {r:i for i,r in enumerate(beta)}
#     x = {r:0 for r in alpha}
#     y = {r:0 for r in beta}

#     # X by alpha
#     for r in alpha:
#         w, h = tiles[r].shape[1], tiles[r].shape[0]
#         preds = [s for s in alpha if pos_b[s] < pos_b[r]]
#         x[r] = 0 if not preds else max(x[s] + tiles[s].shape[1] for s in preds)

#     # Y by beta
#     for r in beta:
#         w, h = tiles[r].shape[1], tiles[r].shape[0]
#         preds = [s for s in beta if pos_a[s] < pos_a[r]]
#         y[r] = 0 if not preds else max(y[s] + tiles[s].shape[0] for s in preds)

#     # Return placement dict
#     placement = {}
#     for r in tiles:
#         w = tiles[r].shape[1]
#         h = tiles[r].shape[0]
#         placement[r] = (x[r], y[r], w, h)
#     return placement

# # --- Count how many desired adjacencies are satisfied ---
# def adjacency_score(placement, graph):
#     score = 0
#     n = graph.shape[0]
#     for i in range(n):
#         for j in range(i+1, n):
#             if graph[i,j]:
#                 xi, yi, wi, hi = placement[i]
#                 xj, yj, wj, hj = placement[j]
#                 Ri = box(xi, yi, xi+wi, yi+hi)
#                 Rj = box(xj, yj, xj+wj, yj+hj)
#                 if Ri.touches(Rj) and Ri.intersection(Rj).length>0:
#                     score += 1
#     return score

# # --- Energy = –adjacencies + heavy penalties for OOB or overlaps ---
# def energy(state, boundary, tiles, graph, PEN=1000):
#     alpha, beta = state
#     placement = decode_sequence_pair(alpha, beta, tiles)
#     E = -adjacency_score(placement, graph)

#     polys = []
#     total_area = 0
#     for r,(x,y,w,h) in placement.items():
#         total_area += w*h
#         # test by center‑point for boundary
#         if not boundary.contains(Point(x + w/2, y + h/2)):
#             E += PEN
#         polys.append(box(x,y,x+w,y+h))

#     U = unary_union(polys)
#     if U.area < total_area - 1e-6:
#         E += PEN * (total_area - U.area)
#     return E

# # --- Simulated Annealing over sequence‑pairs ---
# def optimize(boundary, tiles, graph, max_iter=3000):
#     rooms = list(tiles.keys())
#     alpha, beta = rooms[:], rooms[:]
#     random.shuffle(alpha); random.shuffle(beta)
#     state = (alpha, beta)
#     best, bestE = state, energy(state, boundary, tiles, graph)
#     T = 1.0

#     for _ in range(max_iter):
#         a2, b2 = state[0][:], state[1][:]
#         if random.random()<0.5:
#             i,j = random.sample(rooms,2); a2[i],a2[j]=a2[j],a2[i]
#         else:
#             i,j = random.sample(rooms,2); b2[i],b2[j]=b2[j],b2[i]
#         new = (a2,b2)
#         Enew = energy(new, boundary, tiles, graph)
#         Ecur = energy(state, boundary, tiles, graph)
#         if Enew < Ecur or random.random() < math.exp((Ecur-Enew)/T):
#             state = new
#             if Enew < bestE:
#                 best, bestE = new, Enew
#         T *= 0.995

#     return best, decode_sequence_pair(*best, tiles)

# # --- Matplotlib Visualization ---
# def plot_boundary(boundary, ax):
#     # Draw exterior
#     x,y = boundary.exterior.xy
#     ax.add_patch(MplPolygon(list(zip(x,y)), closed=True, fill=False, edgecolor='k', lw=2))
#     # Draw any holes
#     for hole in boundary.interiors:
#         xi, yi = hole.xy
#         ax.add_patch(MplPolygon(list(zip(xi, yi)), closed=True, fill=False, edgecolor='r', lw=1, ls='--'))

# def plot_rooms(placement, ax):
#     for r,(x,y,w,h) in placement.items():
#         ax.add_patch(Rectangle((x,y), w, h, facecolor='C0', edgecolor='k', alpha=0.5))
#         ax.text(x + w/2, y + h/2, str(r), ha='center', va='center', color='white')

# def visualize(boundary, placement):
#     fig, ax = plt.subplots(figsize=(8,6))
#     plot_boundary(boundary, ax)
#     plot_rooms(placement, ax)
#     ax.set_aspect('equal', 'box')
#     ax.autoscale()
#     ax.set_title("Floorplan in C‑Shaped Boundary")
#     plt.show()

# # --- Main: tie it all together ---
# def main():
#     (rows,cols), boundary, tiles, graph, rotate = processed_data()
#     _, placement = optimize(boundary, tiles, graph)
#     print("Final placements:")
#     for r,(x,y,w,h) in placement.items():
#         print(f" Room {r}: x={x}, y={y}, w={w}, h={h}")
#     visualize(boundary, placement)

# if __name__ == "__main__":
#     main()

# """
# solver.py

# Picks up the in‑code data from fetch_data.py, places each room inside the
# C‑shaped plot, enforces exact adjacency, and visualizes the final layout.
# """

# import random
# import math
# import numpy as np
# from shapely.geometry import box, Point
# from shapely.ops import unary_union
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon as MplPolygon, Rectangle

# from fetch_data import processed_data
# from itertools import combinations

# # -----------------------------------------------------------------------------
# # 1) Sequence‑Pair Decoding
# # -----------------------------------------------------------------------------
# def decode_sequence_pair(alpha, beta, tiles):
#     pos_a = {r:i for i,r in enumerate(alpha)}
#     pos_b = {r:i for i,r in enumerate(beta)}
#     x = {r:0 for r in alpha}
#     y = {r:0 for r in beta}

#     # X-coords by alpha
#     for r in alpha:
#         h, w = tiles[r].shape
#         preds = [s for s in alpha if pos_b[s] < pos_b[r]]
#         x[r] = 0 if not preds else max(x[s] + tiles[s].shape[1] for s in preds)

#     # Y-coords by beta
#     for r in beta:
#         h, w = tiles[r].shape
#         preds = [s for s in beta if pos_a[s] < pos_a[r]]
#         y[r] = 0 if not preds else max(y[s] + tiles[s].shape[0] for s in preds)

#     return {r: (x[r], y[r], tiles[r].shape[1], tiles[r].shape[0]) for r in tiles}

# # -----------------------------------------------------------------------------
# # 2) Enumerate All Touching Pairs
# # -----------------------------------------------------------------------------
# def touching_pairs(placement):
#     touches = set()
#     for i,j in combinations(placement.keys(), 2):
#         xi, yi, wi, hi = placement[i]
#         xj, yj, wj, hj = placement[j]
#         Ri = box(xi, yi, xi+wi, yi+hi)
#         Rj = box(xj, yj, xj+wj, yj+hj)
#         if Ri.touches(Rj) and Ri.intersection(Rj).length>0:
#             touches.add(frozenset({i,j}))
#     return touches

# # -----------------------------------------------------------------------------
# # 3) Energy: missing + extra adjacencies + boundary/overlap penalties
# # -----------------------------------------------------------------------------
# def energy(state, boundary, tiles, graph,
#            w_missing=1000, w_extra=1000, w_penalty=1000):
#     alpha, beta = state
#     placement = decode_sequence_pair(alpha, beta, tiles)

#     # Desired and actual edge sets
#     edges_desired = {frozenset({i,j})
#                      for i in range(graph.shape[0])
#                      for j in range(i+1, graph.shape[1]) if graph[i,j]}
#     edges_actual  = touching_pairs(placement)

#     missing = edges_desired - edges_actual
#     extra   = edges_actual  - edges_desired

#     E = w_missing * len(missing) + w_extra * len(extra)

#     # Boundary and overlap penalties
#     polys = []; total_area = 0
#     for r,(x,y,w,h) in placement.items():
#         total_area += w*h
#         center = Point(x + w/2, y + h/2)
#         if not boundary.contains(center):
#             E += w_penalty
#         polys.append(box(x,y,x+w,y+h))

#     union = unary_union(polys)
#     if union.area < total_area - 1e-6:
#         E += w_penalty * (total_area - union.area)

#     return E

# # -----------------------------------------------------------------------------
# # 4) Simulated Annealing Search
# # -----------------------------------------------------------------------------
# def optimize(boundary, tiles, graph, max_iter=3000):
#     rooms = list(tiles.keys())
#     alpha, beta = rooms[:], rooms[:]
#     random.shuffle(alpha); random.shuffle(beta)
#     state = (alpha, beta)
#     best_state, best_E = state, energy(state, boundary, tiles, graph)

#     T = 1.0
#     for _ in range(max_iter):
#         a2, b2 = state[0][:], state[1][:]
#         if random.random() < 0.5:
#             i,j = random.sample(rooms, 2); a2[i], a2[j] = a2[j], a2[i]
#         else:
#             i,j = random.sample(rooms, 2); b2[i], b2[j] = b2[j], b2[i]

#         new_state = (a2, b2)
#         E_new = energy(new_state, boundary, tiles, graph)
#         E_cur = energy(state, boundary, tiles, graph)

#         if E_new < E_cur or random.random() < math.exp((E_cur - E_new)/T):
#             state = new_state
#             if E_new < best_E:
#                 best_state, best_E = new_state, E_new

#         T *= 0.995

#     final_placement = decode_sequence_pair(*best_state, tiles)
#     return best_state, final_placement

# # -----------------------------------------------------------------------------
# # 5) Matplotlib Visualization
# # -----------------------------------------------------------------------------
# def plot_boundary(boundary, ax):
#     # exterior
#     x,y = boundary.exterior.xy
#     ax.add_patch(MplPolygon(list(zip(x,y)), closed=True,
#                             fill=False, edgecolor='k', lw=2))
#     # holes
#     for hole in boundary.interiors:
#         xi, yi = hole.xy
#         ax.add_patch(MplPolygon(list(zip(xi,yi)), closed=True,
#                                 fill=False, edgecolor='r',
#                                 lw=1, ls='--'))

# def plot_rooms(placement, ax):
#     for r,(x,y,w,h) in placement.items():
#         ax.add_patch(Rectangle((x,y), w, h,
#                                facecolor='C0', edgecolor='k', alpha=0.5))
#         ax.text(x + w/2, y + h/2, str(r),
#                 ha='center', va='center', color='white')

# def visualize(boundary, placement):
#     fig, ax = plt.subplots(figsize=(8,6))
#     plot_boundary(boundary, ax)
#     plot_rooms(placement, ax)
#     ax.set_aspect('equal','box')
#     ax.autoscale()
#     ax.set_title("Exact‑Adjacency Floorplan in C‑Shape")
#     plt.show()

# # -----------------------------------------------------------------------------
# # 6) Main
# # -----------------------------------------------------------------------------
# def main():
#     (rows,cols), boundary, tiles, graph, rotate = processed_data()
#     _, placement = optimize(boundary, tiles, graph)
#     print("Final placement (x,y,w,h):")
#     for r,(x,y,w,h) in placement.items():
#         print(f" Room {r}: x={x:.1f}, y={y:.1f}, w={w}, h={h}")
#     visualize(boundary, placement)

# if __name__ == "__main__":
#     main()

# import random, math
# import numpy as np
# from shapely.geometry import box, Point
# from shapely.ops import unary_union
# from z3 import Optimize, IntVector, And, Or, sat
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon as MplPolygon, Rectangle
# from itertools import combinations

# from fetch_data import processed_data

# def decode_sequence_pair(alpha, beta, tiles):
#     pos_a = {r:i for i,r in enumerate(alpha)}
#     pos_b = {r:i for i,r in enumerate(beta)}
#     x = {r:0 for r in alpha}; y = {r:0 for r in beta}
#     # X by alpha
#     for r in alpha:
#         w = tiles[r].shape[1]
#         preds = [s for s in alpha if pos_b[s] < pos_b[r]]
#         x[r] = 0 if not preds else max(x[s] + tiles[s].shape[1] for s in preds)
#     # Y by beta
#     for r in beta:
#         h = tiles[r].shape[0]
#         preds = [s for s in beta if pos_a[s] < pos_a[r]]
#         y[r] = 0 if not preds else max(y[s] + tiles[s].shape[0] for s in preds)
#     return {r:(x[r], y[r], tiles[r].shape[1], tiles[r].shape[0]) for r in tiles}

# def touching_pairs(placement):
#     touches = set()
#     for i,j in combinations(placement,2):
#         Ri = box(*placement[i]); Rj = box(*placement[j])
#         if Ri.touches(Rj) and Ri.intersection(Rj).length>0:
#             touches.add(frozenset({i,j}))   # 
#     return touches

# def energy(state, boundary, tiles, graph,
#            w_missing=1000, w_extra=1000, w_penalty=1000):
#     alpha,beta = state
#     place = decode_sequence_pair(alpha,beta,tiles)
#     desired = {frozenset({i,j}) for i in range(len(tiles))
#                for j in range(i+1,len(tiles)) if graph[i,j]}
#     actual = touching_pairs(place)
#     missing = desired - actual
#     extra   = actual  - desired
#     E = w_missing*len(missing) + w_extra*len(extra)
#     polys, area = [], 0
#     for r,(x,y,w,h) in place.items():
#         area += w*h
#         center = Point(x+w/2, y+h/2)
#         if not boundary.contains(center): E += w_penalty
#         polys.append(box(x,y,x+w,y+h))
#     U = unary_union(polys)
#     if U.area < area - 1e-6: E += w_penalty*(area-U.area)
#     return E  # 

# def optimize(boundary, tiles, graph, iters=3000):
#     rooms=list(tiles); alpha,beta=rooms[:],rooms[:]
#     random.shuffle(alpha); random.shuffle(beta)
#     state=(alpha,beta); best, bestE=state,energy(state,boundary,tiles,graph)
#     T=1.0
#     for _ in range(iters):
#         a2,b2=state[0][:],state[1][:]
#         if random.random()<0.5:
#             i,j=random.sample(rooms,2); a2[i],a2[j]=a2[j],a2[i]
#         else:
#             i,j=random.sample(rooms,2); b2[i],b2[j]=b2[j],b2[i]
#         new=(a2,b2); Enew=energy(new,boundary,tiles,graph); Ecur=energy(state,boundary,tiles,graph)
#         if Enew<Ecur or random.random()<math.exp((Ecur-Enew)/T):
#             state=new
#             if Enew<bestE: best,bestE=state,Enew
#         T*=0.995
#     return best, decode_sequence_pair(*best,tiles)

# def plot_boundary(boundary,ax):
#     x,y=boundary.exterior.xy
#     ax.add_patch(MplPolygon(list(zip(x,y)),closed=True,fill=False,edgecolor='k',lw=2))
#     for hole in boundary.interiors:
#         xi,yi=hole.xy
#         ax.add_patch(MplPolygon(list(zip(xi,yi)),closed=True,fill=False,edgecolor='r',lw=1,ls='--'))

# def plot_rooms(place,ax):
#     for r,(x,y,w,h) in place.items():
#         ax.add_patch(Rectangle((x,y),w,h,facecolor='C0',edgecolor='k',alpha=0.5))
#         ax.text(x+w/2,y+h/2,str(r),ha='center',va='center',color='white')

# def visualize(boundary,place):
#     fig,ax=plt.subplots(figsize=(8,6))
#     plot_boundary(boundary,ax); plot_rooms(place,ax)
#     ax.set_aspect('equal','box'); ax.autoscale(); ax.set_title("Exact Adjacency")
#     plt.show()  # 

# def main():
#     (rows,cols),boundary,tiles,graph,rotate=processed_data()
#     _,placement=optimize(boundary,tiles,graph)
#     print("Final placement:",placement)
#     visualize(boundary,placement)

# if __name__=="__main__":
#     main()

# import random, math
# import numpy as np
# from shapely.geometry import box, Point
# from shapely.ops import unary_union
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon as MplPolygon, Rectangle
# from fetch_data import processed_data
# from itertools import combinations

# # 1) Sequence‑Pair decoding
# def decode_sequence_pair(alpha, beta, tiles):
#     pos_a = {r:i for i,r in enumerate(alpha)}
#     pos_b = {r:i for i,r in enumerate(beta)}
#     x = {r:0 for r in alpha}
#     y = {r:0 for r in beta}

#     for r in alpha:
#         preds = [s for s in alpha if pos_b[s] < pos_b[r]]
#         x[r] = 0 if not preds else max(x[s] + tiles[s].shape[1] for s in preds)
#     for r in beta:
#         preds = [s for s in beta if pos_a[s] < pos_a[r]]
#         y[r] = 0 if not preds else max(y[s] + tiles[s].shape[0] for s in preds)

#     return {r: (x[r], y[r], tiles[r].shape[1], tiles[r].shape[0]) for r in tiles}

# # 2) Find all touching pairs
# def touching_pairs(placement):
#     touches = set()
#     for i,j in combinations(placement, 2):
#         xi, yi, wi, hi = placement[i]
#         xj, yj, wj, hj = placement[j]
#         Ri = box(xi, yi, xi+wi, yi+hi)
#         Rj = box(xj, yj, xj+wj, yj+hj)
#         if Ri.touches(Rj) and Ri.intersection(Rj).length > 0:
#             touches.add(frozenset({i,j}))
#     return touches

# # 3) Energy with exact adjacency enforcement
# def energy(state, boundary, tiles, graph,
#            w_missing=1000, w_extra=1000, w_penalty=1000):
#     alpha, beta = state
#     placement = decode_sequence_pair(alpha, beta, tiles)

#     # Desired vs actual edges
#     edges_desired = {frozenset({i,j})
#                      for i in range(graph.shape[0])
#                      for j in range(i+1, graph.shape[1]) if graph[i,j]}
#     edges_actual = touching_pairs(placement)

#     missing = edges_desired - edges_actual
#     extra   = edges_actual  - edges_desired

#     E = w_missing*len(missing) + w_extra*len(extra)

#     # Boundary & overlap penalties
#     polys = []; total = 0
#     for r,(x,y,w,h) in placement.items():
#         total += w*h
#         center = Point(x + w/2, y + h/2)
#         if not boundary.contains(center):
#             E += w_penalty
#         polys.append(box(x,y,x+w,y+h))

#     U = unary_union(polys)
#     if U.area < total - 1e-6:
#         E += w_penalty * (total - U.area)

#     return E

# # 4) Simulated Annealing optimizer
# def optimize(boundary, tiles, graph, max_iter=3000):
#     rooms = list(tiles.keys())
#     alpha, beta = rooms[:], rooms[:]
#     random.shuffle(alpha); random.shuffle(beta)
#     state = (alpha, beta)
#     best, bestE = state, energy(state, boundary, tiles, graph)
#     T = 1.0

#     for _ in range(max_iter):
#         a2, b2 = state[0][:], state[1][:]
#         if random.random() < 0.5:
#             i,j = random.sample(rooms, 2); a2[i],a2[j] = a2[j],a2[i]
#         else:
#             i,j = random.sample(rooms, 2); b2[i],b2[j] = b2[j],b2[i]

#         new_state = (a2, b2)
#         E_new = energy(new_state, boundary, tiles, graph)
#         E_cur = energy(state, boundary, tiles, graph)

#         if E_new < E_cur or random.random() < math.exp((E_cur - E_new)/T):
#             state = new_state
#             if E_new < bestE:
#                 best, bestE = new_state, E_new

#         T *= 0.995

#     final_place = decode_sequence_pair(*best, tiles)
#     return best, final_place

# # 5) Visualization
# def plot_boundary(boundary, ax):
#     x, y = boundary.exterior.xy
#     ax.add_patch(MplPolygon(list(zip(x, y)), closed=True,
#                             fill=False, edgecolor='k', linewidth=2))
#     for hole in boundary.interiors:
#         xi, yi = hole.xy
#         ax.add_patch(MplPolygon(list(zip(xi, yi)), closed=True,
#                                 fill=False, edgecolor='r',
#                                 linewidth=1, linestyle='--'))

# def plot_rooms(placement, ax):
#     for r,(x,y,w,h) in placement.items():
#         ax.add_patch(
#             Rectangle((x, y), w, h, facecolor='C0', edgecolor='k', alpha=0.5)
#         )
#         ax.text(x + w/2, y + h/2, str(r),
#                 ha='center', va='center', color='white')

# def visualize(boundary, placement):
#     fig, ax = plt.subplots(figsize=(8,6))
#     plot_boundary(boundary, ax)
#     plot_rooms(placement, ax)
#     ax.set_aspect('equal', 'box')
#     ax.autoscale()
#     ax.set_title("Exact‑Adjacency Floorplan in C‑Shape")
#     plt.show()

# # 6) Main entry
# def main():
#     (rows, cols), boundary, tiles, graph, rotate = processed_data()
#     _, placement = optimize(boundary, tiles, graph)
#     print("Final placement:")
#     for r,(x,y,w,h) in placement.items():
#         print(f" Room {r}: x={x:.1f}, y={y:.1f}, w={w}, h={h}")
#     visualize(boundary, placement)

# if __name__ == "__main__":
#     main()

# import random
# import math
# import numpy as np
# from shapely.geometry import box, Point, LinearRing, MultiPolygon
# from shapely.ops import unary_union
# from fetch_data import processed_data
# from itertools import combinations

# # 1) Sequence‑Pair decoding
# # unchanged (as before)
# def decode_sequence_pair(alpha, beta, tiles):
#     pos_a = {r:i for i,r in enumerate(alpha)}
#     pos_b = {r:i for i,r in enumerate(beta)}
#     x = {r:0 for r in alpha}
#     y = {r:0 for r in beta}

#     for r in alpha:
#         preds = [s for s in alpha if pos_b[s] < pos_b[r]]
#         x[r] = 0 if not preds else max(x[s] + tiles[s].shape[1] for s in preds)
#     for r in beta:
#         preds = [s for s in beta if pos_a[s] < pos_a[r]]
#         y[r] = 0 if not preds else max(y[s] + tiles[s].shape[0] for s in preds)

#     return {r: (x[r], y[r], tiles[r].shape[1], tiles[r].shape[0]) for r in tiles}

# # 2) Find all touching pairs
# # unchanged (as before)
# def touching_pairs(placement):
#     touches = set()
#     for i,j in combinations(placement, 2):
#         xi, yi, wi, hi = placement[i]
#         xj, yj, wj, hj = placement[j]
#         Ri = box(xi, yi, xi+wi, yi+hi)
#         Rj = box(xj, yj, xj+wj, yj+hj)
#         if Ri.touches(Rj) and Ri.intersection(Rj).length > 0:
#             touches.add(frozenset({i,j}))
#     return touches

# # 3) Count concave corners (bends) of the union of rooms

# def count_bends(placement):
#     """
#     Count concave corners (interior angles >180°) in the exterior boundary
#     of the unified room placement. Handles both Polygon and MultiPolygon.
#     """
#     polys = [box(x, y, x+w, y+h) for (x,y,w,h) in placement.values()]
#     U = unary_union(polys)

#     # Collect all exterior rings
#     if isinstance(U, MultiPolygon):
#         rings = [LinearRing(poly.exterior.coords) for poly in U.geoms]
#     else:
#         rings = [LinearRing(U.exterior.coords)]

#     bends = 0
#     for ring in rings:
#         coords = list(ring.coords)
#         for i in range(len(coords)):
#             p_prev = coords[i-1]
#             p = coords[i]
#             p_next = coords[(i+1) % len(coords)]
#             v1 = (p_prev[0] - p[0], p_prev[1] - p[1])
#             v2 = (p_next[0] - p[0], p_next[1] - p[1])
#             cross = v1[0]*v2[1] - v1[1]*v2[0]
#             # Negative cross indicates interior angle > 180° (concave)
#             if cross < 0:
#                 bends += 1
#     return bends

# # 4) Energy with adjacency, boundary, overlap, and bend penalties
# def energy(state, boundary, tiles, graph,
#            w_missing=1000, w_extra=1000, w_penalty=1000, w_bend=500):
#     alpha, beta = state
#     placement = decode_sequence_pair(alpha, beta, tiles)

#     # Desired vs actual adjacency
#     edges_desired = {frozenset({i,j})
#                      for i in range(graph.shape[0])
#                      for j in range(i+1, graph.shape[1]) if graph[i,j]}
#     edges_actual = touching_pairs(placement)
#     missing = edges_desired - edges_actual
#     extra   = edges_actual  - edges_desired
#     E = w_missing*len(missing) + w_extra*len(extra)

#     # Boundary & overlap penalties
#     polys = []
#     total = 0
#     for r,(x,y,w,h) in placement.items():
#         total += w*h
#         center = Point(x + w/2, y + h/2)
#         if not boundary.contains(center):
#             E += w_penalty
#         polys.append(box(x,y,x+w,y+h))

#     U = unary_union(polys)
#     if U.area < total - 1e-6:
#         E += w_penalty * (total - U.area)

#     # Bend penalty
#     bends = count_bends(placement)
#     E += w_bend * bends

#     return E

# # 5) Simulated Annealing optimizer (updated to pass w_bend)
# def optimize(boundary, tiles, graph, max_iter=3000,
#              w_missing=1000, w_extra=1000, w_penalty=1000, w_bend=500):
#     rooms = list(tiles.keys())
#     alpha, beta = rooms[:], rooms[:]
#     random.shuffle(alpha); random.shuffle(beta)
#     state = (alpha, beta)
#     best, bestE = state, energy(state, boundary, tiles, graph,
#                                  w_missing, w_extra, w_penalty, w_bend)
#     T = 1.0

#     for _ in range(max_iter):
#         a2, b2 = state[0][:], state[1][:]
#         if random.random() < 0.5:
#             i,j = random.sample(rooms, 2); a2[i],a2[j] = a2[j],a2[i]
#         else:
#             i,j = random.sample(rooms, 2); b2[i],b2[j] = b2[j],b2[i]

#         new_state = (a2, b2)
#         E_new = energy(new_state, boundary, tiles, graph,
#                        w_missing, w_extra, w_penalty, w_bend)
#         E_cur = energy(state, boundary, tiles, graph,
#                        w_missing, w_extra, w_penalty, w_bend)

#         if E_new < E_cur or random.random() < math.exp((E_cur - E_new)/T):
#             state = new_state
#             if E_new < bestE:
#                 best, bestE = new_state, E_new
#         T *= 0.995

#     final_place = decode_sequence_pair(*best, tiles)
#     return best, final_place

# # 6) Visualization unchanged
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon as MplPolygon, Rectangle

# def plot_boundary(boundary, ax):
#     x, y = boundary.exterior.xy
#     ax.add_patch(MplPolygon(list(zip(x, y)), closed=True,
#                             fill=False, edgecolor='k', linewidth=2))
#     for hole in boundary.interiors:
#         xi, yi = hole.xy
#         ax.add_patch(MplPolygon(list(zip(xi, yi)), closed=True,
#                                 fill=False, edgecolor='r',
#                                 linewidth=1, linestyle='--'))

# def plot_rooms(placement, ax):
#     for r,(x,y,w,h) in placement.items():
#         ax.add_patch(
#             Rectangle((x, y), w, h, facecolor='C0', edgecolor='k', alpha=0.5)
#         )
#         ax.text(x + w/2, y + h/2, str(r),
#                 ha='center', va='center', color='white')

# def visualize(boundary, placement):
#     fig, ax = plt.subplots(figsize=(8,6))
#     plot_boundary(boundary, ax)
#     plot_rooms(placement, ax)
#     ax.set_aspect('equal', 'box')
#     ax.autoscale()
#     ax.set_title("Floorplan optimizing adjacencies and bends")
#     plt.show()

# # 7) Main entry updated
# if __name__ == "__main__":
#     (rows, cols), boundary, tiles, graph, rotate = processed_data()
#     _, placement = optimize(boundary, tiles, graph,
#                              max_iter=5000, w_bend=200)
#     print("Final placement (including bend minimization):")
#     for r,(x,y,w,h) in placement.items():
#         print(f" Room {r}: x={x:.1f}, y={y:.1f}, w={w}, h={h}")
#     visualize(boundary, placement)

import random
import math
import numpy as np
from shapely.geometry import box, Point, LinearRing, MultiPolygon
from shapely.ops import unary_union
from fetch_data import processed_data

# 1) Sequence‑Pair decoding
def decode_sequence_pair(alpha, beta, tiles):
    pos_a = {r:i for i,r in enumerate(alpha)}
    pos_b = {r:i for i,r in enumerate(beta)}
    x = {r:0 for r in alpha}
    y = {r:0 for r in beta}

    # horizontal ordering from beta-ranks
    for r in alpha:
        preds = [s for s in alpha if pos_b[s] < pos_b[r]]
        x[r] = 0 if not preds else max(x[s] + tiles[s].shape[1] for s in preds)
    # vertical ordering from alpha-ranks
    for r in beta:
        preds = [s for s in beta if pos_a[s] < pos_a[r]]
        y[r] = 0 if not preds else max(y[s] + tiles[s].shape[0] for s in preds)

    return {r: (x[r], y[r], tiles[r].shape[1], tiles[r].shape[0]) for r in tiles}

# 2) Touching pairs detection
def touching_pairs(placement):
    touches = set()
    for i in placement:
        xi, yi, wi, hi = placement[i]
        Ri = box(xi, yi, xi+wi, yi+hi)
        for j in placement:
            if j <= i: continue
            xj, yj, wj, hj = placement[j]
            Rj = box(xj, yj, xj+wj, yj+hj)
            if Ri.touches(Rj) and Ri.intersection(Rj).length > 0:
                touches.add(frozenset({i,j}))
    return touches

# 3) Bend counting (optional)
def count_bends(placement):
    polys = [box(x, y, x+w, y+h) for (x,y,w,h) in placement.values()]
    U = unary_union(polys)
    rings = (
        [LinearRing(poly.exterior.coords) for poly in U.geoms]
        if isinstance(U, MultiPolygon)
        else [LinearRing(U.exterior.coords)]
    )
    bends = 0
    for ring in rings:
        coords = list(ring.coords)
        for i in range(len(coords)):
            p_prev, p, p_next = coords[i-1], coords[i], coords[(i+1)%len(coords)]
            v1 = (p_prev[0]-p[0], p_prev[1]-p[1])
            v2 = (p_next[0]-p[0], p_next[1]-p[1])
            if v1[0]*v2[1] - v1[1]*v2[0] < 0:
                bends += 1
    return bends

# 4) Energy function
def energy(state, boundary, tiles, graph,
           w_missing=1000, w_extra=1000, w_dev=10, w_bend=50):
    alpha, beta, sizes = state
    placement = decode_sequence_pair(alpha, beta, tiles)

    # adjacency
    edges_desired = {frozenset({i,j}) for i in range(graph.shape[0])
                     for j in range(i+1, graph.shape[1]) if graph[i,j]}
    edges_actual  = touching_pairs(placement)
    missing = edges_desired - edges_actual
    extra   = edges_actual - edges_desired

    E = w_missing*len(missing) + w_extra*len(extra)

    # size deviation (here skip, as sizes fixed)
    # bends penalty
    E += w_bend * count_bends(placement)

    # boundary containment & overlap
    total_area = 0
    polys = []
    for r,(x,y,w,h) in placement.items():
        total_area += w*h
        center = Point(x + w/2, y + h/2)
        if not boundary.contains(center): E += w_dev
        polys.append(box(x,y,x+w,y+h))
    U = unary_union(polys)
    if U.area < total_area - 1e-6:
        E += w_dev * (total_area - U.area)

    return E

# 5) Simulated Annealing solver
def optimize(boundary, tiles, graph, max_iter=2000):
    n = len(tiles)
    rooms = list(tiles.keys())
    alpha = rooms[:]; beta = rooms[:]
    random.shuffle(alpha); random.shuffle(beta)
    best_state = (alpha[:], beta[:], None)
    bestE = energy((alpha,beta,None), boundary, tiles, graph)
    T = 1.0

    for it in range(max_iter):
        # neighbor: swap in alpha or beta
        a2, b2 = alpha[:], beta[:]
        if random.random() < 0.5:
            i,j = random.sample(rooms, 2); a2[i],a2[j] = a2[j],a2[i]
        else:
            i,j = random.sample(rooms, 2); b2[i],b2[j] = b2[j],b2[i]

        E2 = energy((a2,b2,None), boundary, tiles, graph)
        if E2 < bestE or random.random() < math.exp((bestE-E2)/T):
            alpha, beta = a2, b2
            bestE = E2
            best_state = (alpha[:], beta[:], None)
        T *= 0.995

    return decode_sequence_pair(best_state[0], best_state[1], tiles)

