import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle

def plot_floorplan(boundary, placement, title="C‑shaped Floorplan"):
    fig, ax = plt.subplots(figsize=(6,5))
    # boundary
    bx, by = boundary.exterior.xy
    ax.add_patch(MplPolygon(list(zip(bx,by)), closed=True,
                            fill=False, edgecolor='black', linewidth=2))
    for hole in boundary.interiors:
        hx, hy = hole.xy
        ax.add_patch(MplPolygon(list(zip(hx,hy)), closed=True,
                                fill=False, edgecolor='red', linestyle='--'))
    # rooms
    for r,(x,y,w,h) in placement.items():
        ax.add_patch(Rectangle((x,y), w, h,
                    facecolor='C0', alpha=0.5, edgecolor='black'))
        ax.text(x+w/2, y+h/2, str(r), color='white',
                ha='center', va='center', fontsize=12)
    ax.set_aspect('equal', 'box')
    ax.set_title(title)

    # ensure all patches are within view
    ax.relim()
    ax.autoscale_view()
    
    plt.axis('off')
    plt.show()