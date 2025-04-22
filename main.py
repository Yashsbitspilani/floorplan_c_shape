from fetch_data import processed_data
from solver import optimize
from visualisation import plot_floorplan

if __name__ == "__main__":
    (rows, cols), boundary, tiles, graph, rotate = processed_data()
    placement = optimize(boundary, tiles, graph, max_iter=5000)
    plot_floorplan(boundary, placement)