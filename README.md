# C‑Shaped Floorplan Sequence‑Pair Solver

## Overview  
This project generates and visualizes a floorplan within a C‑shaped boundary using a sequence‑pair representation and simulated annealing.

## Repository Structure
- `main.py`            – Entry point: reads data, runs the optimizer, writes output.
- `fetch_data.py`      – Supplies the C‑shape boundary, tile shapes, and adjacency graph.
- `solver.py`          – Implements sequence‑pair decoding, energy function, and annealer.
- `visualisation.py`   – Plotting routines for boundary and room placement.
- `requirements.txt`   – Python dependencies.

## Prerequisites
- Python 3.7+
- `numpy`
- `shapely`
- `matplotlib`

Install dependencies via:
```bash
pip install -r requirements.txt
