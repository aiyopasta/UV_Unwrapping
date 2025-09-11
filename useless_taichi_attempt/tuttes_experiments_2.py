# Well, well, well... they say the dog returns to its original owner.
# (I'll remake this reel —— better script, realtime code, better results.)
#
# This is just the experiments file where I try getting it to work in realtime in Taichi.

# OHHHH MY GOD THIS WAS A WASTE OF TIME THE OTHER CODE WOULD'VE WORKED IN REAL-TIME IF YOU'D JUST DEFINED THE
# MATRIX U USED FOR THE SOLVE OUTSIDE THE MAIN WHILE LOOP!!!

import time
import taichi as ti
import taichi.math as tm
import numpy as np
import scipy as sp  # for SPARSE matrices
import seaborn as sns  # for epic colormaps
import trimesh  # for easily loading in a mesh
import sys
np.set_printoptions(threshold=np.inf, suppress=True)
sys.tracebacklimit=0

# Taichi setup
# ti.init(arch=ti.cpu, cpu_max_num_threads=1)  # can switch to ti.cpu and thread=1 for debugging things in serial!
ti.init(arch=ti.gpu)
if ti.cfg.arch == ti.metal:
    print("Using Metal backend with GPU ✅")
else:
    print("Not using Metal backend ❌")

# Width, height of full window
scale = 0.5  # 0.7 on full screen
width, height = int(1920 * scale), int(1920 * scale)
wh = ti.Vector([width, height])
# Screen pixel array & gui object
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(width, height))
window = ti.ui.Window("Taichi Path-tracing!", (width, height), vsync=True)
canvas = window.get_canvas()


# The usual coordinate shift & inverse (NOTE: ti.types.vector is like np.ndarray!)
# NOTE: In Taichi, +y is UPWARDS and (0, 0) is BOTTOM LEFT!!

# The non-taichi-fied coordinate shifts.
# Eat positions in the (0,0)-botleft, y-upwards, [0, 1] range.
def A_(pt: np.ndarray):
    return (pt + np.array([width / 2, height / 2])) / np.array([width, height])

def A_inv_(pt: np.ndarray):
    return (pt - np.array([width / 2, height / 2])) * np.array([width, height])

# Generic mat-mul function for Taichi


# Just if you wanna paint things pixel-by-pixel
@ti.kernel
def paint():
    # Clear frame
    pixels.fill(0)

# Mesh info
filename = '/Users/adityaabhyankar/Desktop/Programming/UV_Unwrapping_2/meshes/THE_ACTUAL_FACE_TO_USE.obj'

# Load OBJ file using trimesh
mesh = trimesh.load('/Users/adityaabhyankar/Desktop/Programming/UV_Unwrapping_2/meshes/THE_ACTUAL_FACE_TO_USE.obj')
faces = mesh.faces
edges = mesh.edges_unique

# Set some verts as the "pinned" ones
pinned = [128, 418, 142, 417, 143, 412, 149, 409, 134, 425, 133, 422, 137, 407, 147, 414, 146, 410, 145, 415, 144, 408, 141, 419, 140, 416, 136, 423, 135, 424, 132, 426, 131, 421, 138, 420, 139, 413, 148, 411, 130, 598, 427, 129, 428]

# Set positions of non-pinned verts to be 2D, and randomized within a square of side x% of the width of window,
# and override positions of pinned verts to be 2D, and in a circle of diameter y% of the width of the window, y > x.
diam = width * 0.9
side = np.sqrt(2.0) * (width / 2.0 * 0.8)
verts = np.random.uniform(-side/2, side/2, size=(len(mesh.vertices), 2))
for idx in range(len(pinned)):
    theta = (idx / len(pinned)) * 2.0 * np.pi
    verts[pinned[idx]] = (diam / 2.0) * np.array([np.cos(theta), np.sin(theta)])


# Additional vars
colors = {
    'white': np.array([255., 255., 255.]),
    'black': np.array([0., 0., 0.]),
    'red': np.array([255, 66, 48]),
    'blue': np.array([30, 5, 252]),
    'fullred': np.array([255, 0, 0]),
    'fullblue': np.array([0, 0, 255]),
    'START': np.array([255, 255, 255])
}

# Create taichi fields out of the vertex positions and edges list for simulation & drawing
verts_ti = ti.Vector.field(2, dtype=ti.f32, shape=verts.shape[0])
verts_ti.from_numpy(np.array(verts, dtype=np.float32))



# SUPER UGLY annoying machinery for drawing (Taichi 🙄)
unpinned_buff = ti.Vector.field(2, dtype=ti.f32, shape=verts.shape[0] - len(pinned))
pinned_buff = ti.Vector.field(2, dtype=ti.f32, shape=len(pinned))
edge_verts_ti  = ti.Vector.field(2, ti.f32, shape=2*edges.shape[0])
edge_colors_ti = ti.Vector.field(3, ti.f32, shape=2*edges.shape[0])
def ugly():
    t = (np.arange(edges.shape[0], dtype=np.float32) / float(edges.shape[0]))[:, None]  # 0 .. 1-1/E
    blue = colors['blue'][None, :].astype(np.float32)
    red  = colors['red'][None, :].astype(np.float32)
    edge_rgb_255 = (1.0 - t) * blue + t * red               # blue -> red
    edge_rgb_255 = np.clip(np.rint(edge_rgb_255), 0, 255).astype(np.uint8)
    edge_colors  = (edge_rgb_255.astype(np.float32) / 255.0)  # -> [0,1]
    edge_colors_v = np.repeat(edge_colors, 2, axis=0)
    verts_norm = A_(verts).astype(np.float32)
    edge_verts = verts_norm[edges.reshape(-1)]
    edge_verts_ti.from_numpy(edge_verts)
    edge_colors_ti.from_numpy(edge_colors_v)
ugly()


# Main window loop
frame = 0
while window.running:
    start = time.time()
    # Paint to screen
    paint()
    canvas.set_image(pixels)

    # Draw stuff using simple draw commands...
    # (1) Draw the edges
    canvas.lines(edge_verts_ti, width=0.0018, indices=None, per_vertex_color=edge_colors_ti)
    # (2) Draw the vertices (unpinned + pinned)
    rad = 0.003  # (it's a fraction of the window width——ya, it's weird)
    unpinned_buff.from_numpy(np.array(A_(np.delete(verts_ti.to_numpy(), pinned, axis=0)), dtype=np.float32))
    canvas.circles(unpinned_buff, rad, color=(1.0, 1.0, 1.0))
    pinned_buff.from_numpy(np.array(A_(verts_ti.to_numpy())[pinned], dtype=np.float32))
    canvas.circles(pinned_buff, rad, color=(1.0, 1.0, 1.0))



    window.show()  # call this AFTER all types of drawing are finished

    # Handle escape key for quitting window
    for e in window.get_events(ti.ui.RELEASE):
        if e.key == ti.ui.ESCAPE:
            window.running = False

