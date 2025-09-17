import copy

import numpy as np
import pygame
import os
import trimesh
import scipy as sp
# import taichi as ti  # for animating nodes fast in a compute kernel
# import taichi.math as tm
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)
np.random.seed(42)  # fix a seed for the same random values

# # Taichi (ONLY for compute kernels)
# # ti.init(arch=ti.cpu, cpu_max_num_threads=1)  # can switch to ti.cpu and thread=1 for debugging things in serial!
# ti.init(arch=ti.gpu)
# if ti.cfg.arch == ti.metal:
#     print("Using Metal backend with GPU ✅")
# else:
#     print("Not using Metal backend ❌")

# Save the animation? TODO: Make sure you're saving to correct destination!!
save_anim = False

# Pygame + gameloop setup
scale = 0.7
width = 1080 * scale
height = 1920 * scale
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Tutte's Embedding")
pygame.init()

# Will always return the same rng given input (useful for initial swirling animation)
def deterministic_rand(x):
    rng = np.random.default_rng(seed=hash(x) % (2**32))
    return rng.random()


# Drawing Coordinate Shift Functions
def A(val):
    return np.array([val[0] + width / 2, -val[1] + height / 2])


def A_inv(val):
    global width, height
    return np.array([val[0] - width / 2, -(val[1] - height / 2)])


def A_many(vals):
    return [A(v) for v in vals]   # Galaxy brain function


# Helper Functions
def lerp(u, initial, final):
    return ((1. - u) * initial) + (u * final)

def squash(t_, intervals=None):
    global keys
    if intervals is None:
        intervals = keys
    for i in range(len(intervals) - 1):
        if intervals[i] <= t_ < intervals[i + 1]:
            return (t_ - intervals[i]) / (intervals[i + 1] - intervals[i]), i

    return intervals[-1], len(intervals) - 2


# Specific case of the squash. We squash t into equally sized intervals.
def squash2(t_, n_intervals=1):
    intervals = [float(i) / n_intervals for i in range(n_intervals + 1)]
    return squash(t_, intervals)


# For rectangular boundary
def rect(t_, rect_length, rect_height):
    verts = [np.array([-rect_length/2.0, +rect_height/2.0]),
             np.array([+rect_length/2.0, +rect_height/2.0]),
             np.array([+rect_length/2.0, -rect_height/2.0]),
             np.array([-rect_length/2.0, -rect_height/2.0])]  # starting with top-left, counter-clockwise

    tau, side = squash2(t_, 4)
    tau = 1 if abs(tau - 1) < 0.05 else tau  # clean corners
    v0, v1 = verts[side], verts[(side + 1) % 4]
    return lerp(tau, v0, v1)


# Rotation matrix
def rot_mat(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s],
                     [s,  c]])


# Squeeze actual interpolation to be within [new_start, new_end], and make it 0 and 1 outside this range
def slash(t_, new_start=0.0, new_end=1.0):
    if t_ < new_start:
        return 0.0

    if t_ > new_end:
        return 1.0

    return (t_ - new_start) / (new_end - new_start)


# Easing functions.
# TODO: Add more!
def ease_inout(t_):
    return np.power(t_, 2) / (np.power(t_, 2) + np.power(1 - t_, 2))


def ease_inout2(t_, beta, center=0.5):
    r = -np.log(2) / np.log(center)
    power = np.power(t_, r)
    return 1.0 / (1.0 + np.power(power / (1 - power), -beta)) if t_ not in [0., 1., 0, 1] else t_


def ease_out(t_):
    return 1.0 - np.power(1.0 - t_, 2.0)


def blink_bump(t_, p):
    assert 0.05 <= p <= 0.95
    return np.exp(-100. * np.power(t_ - p, 2.0) / np.power(0.1, 2.0))


# Inverse of easing functions
# TODO: Add more!
def ease_inout_inverse(t_):
    return (t_ - np.sqrt((1 - t_) * t_)) / ((2 * t_) - 1)


# Mesh information
mesh_file = 'meshes/CUT_CAESAR_MASK_CUT_HOLES.obj'  # Pins below: (Main, eye, mouth, nose)
main_boundary = [92, 1188, 3509, 3495, 1507, 3531, 3527, 3505, 3459, 3544, 3537, 3478, 3463, 2614, 3488, 3494, 3432, 3460, 3462, 3102, 3228, 2561, 3468, 3520, 3490, 3500, 3511, 3485, 3503, 2059, 1443, 1638, 2302, 3471, 3524, 3461, 3499, 3199, 2714, 2604, 3512, 3186, 3164, 3163, 2944, 3540, 3466, 3474, 2027, 1334, 3538, 925, 3552, 3498, 3481, 3355, 3479, 2177, 3484, 3546, 3123, 3469, 3482, 2016, 3457, 3455, 3539, 3525, 3515, 2970, 2969, 3519, 3510, 3476, 3480, 759, 3534, 3542, 3526, 3493, 2489, 3496, 3506, 3124, 3504, 3523, 3501, 2923, 3470, 3477, 3497, 3456, 3548, 3458, 2583, 3475, 3535, 3533, 2729, 2061, 3343, 3491, 3213, 2187, 3553, 3502, 3532, 3489, 3465, 3483, 3545, 3547, 3541, 3473, 3530, 3150, 3467, 3528, 3550, 3551, 1465, 3521, 3507, 3514, 3508, 3522, 3543, 2054, 3516, 3472, 3038, 1599, 3487, 2082, 3513, 3536, 3529, 1695, 3464, 2286, 3549, 3156, 3486, 2895, 3517, 3518, 3492, 2452]
eye = [210, 2447, 985, 1575, 1418, 1319, 872, 843, 262, 333, 334, 834, 467, 468, 259, 260, 578, 579, 886, 887, 711, 296, 298, 775]
mouth = [54, 55, 3594, 1175, 3599, 3584, 589, 148, 3592, 3575, 3574, 3600, 662, 1111, 3585, 569, 878, 483, 695, 3578, 3579, 588, 131, 3593, 1177, 3583, 3582, 3588, 3595, 3577, 3576, 149, 696, 3580, 153, 3573, 3572, 281, 280, 3591, 3598, 2395, 1752, 1753, 3586, 3587, 388, 1360, 3590, 3597, 190, 673, 3581, 152, 3596, 3589, 724, 443, 1020, 771]
nose = [3554, 3559, 3555, 3560, 3558, 3557, 3562, 3556, 3561, 3563]
# Simply for COLORING purposes. These are the ones we'd like to color in differently
special_verts = set(sum([main_boundary], []))  # set for O(1) look-up
# Set initial pinned (doesn't matter at this point though)
pinned_list = main_boundary
pinned_set = set(main_boundary)

stiff = 2000
h = 0.01

mesh = trimesh.load(mesh_file)
faces = mesh.faces
edges = mesh.edges_unique
positions = np.zeros((len(mesh.vertices), 2))

# 2. Override positions with randomized values
for i in range(positions.shape[0]):
    x_max, y_max = width * 0.9, width * 0.9
    positions[i] = np.random.rand(2) * [x_max, y_max] - [x_max / 2, y_max / 2]

# 3. Initialize velocities to be 0
velocities = np.zeros((len(positions), 2))

# 4. Initialize 2|E| x 2|V| sized connections matrix, as a sparse coo matrix
# (a) First, gather the sparse data. For each entry in the matrix, we store indices (row, col, data).
rows, cols, data = [], [], []
for row, (i1, i2) in enumerate(edges):
    # x-dimension constraint
    rows += [2 * row, 2 * row]
    cols += [2 * i1, 2 * i2]
    data += [1, -1]

    # y-dimension constraint
    rows += [(2 * row) + 1, (2 * row) + 1]
    cols += [(2 * i1) + 1, (2 * i2) + 1]
    data += [1, -1]
# (b) Create the coo matrix, and convert it to csr
C = sp.sparse.coo_matrix((data, (rows, cols)), shape=(2*len(edges), 2*len(positions))).tocsr()

# 5. Compute the sparse inverse matrix operator A^{-1} = (I + alpha * CTC)^-1 in csc format (good for factorization)
CTC = C.T @ C
D = sp.sparse.identity(CTC.shape[0], format='csc') + (h * h * stiff) * CTC
D_inv_op = sp.sparse.linalg.factorized(D)  # note: this is an OPERATOR, not an explicit matrix!

# 6. Compute sparse solver matrix as a linear operator (also not an explicit matrix)
M = sp.sparse.linalg.LinearOperator((2 * len(positions), 2 * len(positions)),
                                    matvec=lambda x: -h * stiff * D_inv_op(CTC @ x))

# Additional vars
colors = {
    'white': np.array([255., 255., 255.]),
    'black': np.array([0., 0., 0.]),
    'red': np.array([255., 66., 48.]),
    'blue': np.array([30., 5., 252.]),
    'fullred': np.array([255., 0., 0.]),
    'fullblue': np.array([0., 0., 255.]),
    'START': np.array([255., 255., 255.]),
    'darkblue': np.array([0., 122., 255.]),
    'lightblue': np.array([10., 132., 255.]),
    'darkgreen': np.array([52., 199., 89.]),
    'lightgreen': np.array([48., 209., 88.]),
    'darkindigo': np.array([88., 86., 214.]),
    'lightindigo': np.array([94., 92., 230.]),
    'darkorange': np.array([255., 149., 0.]),
    'lightorange': np.array([255., 159., 10.]),
    'darkpink': np.array([255., 45., 85.]),
    'lightpink': np.array([255., 55., 95.]),
    'darkpurple': np.array([175., 82., 222.]),
    'lightpurple': np.array([191., 90., 242.]),
    'darkred': np.array([255., 59., 48.]),
    'lightred': np.array([255., 69., 58.]),
    'darkteal': np.array([90., 200., 250.]),
    'lightteal': np.array([100., 210., 255.]),
    'darkyellow': np.array([255., 204., 0.]),
    'lightyellow': np.array([255., 214., 10.])
}



# Keyhandling (for zooming, moving, and rotation)
calm, delta, theta_delta = 1.02, 20.0, 0.05
def handle_keys(keys_pressed):
    global positions, calm, pinned_list
    assert calm >= 1.0, 'calm aint so calm!'

    # For zooming in / out
    if keys_pressed[pygame.K_p]:
        positions *= calm
    elif keys_pressed[pygame.K_o]:
        positions /= calm

    # For moving about
    if keys_pressed[pygame.K_w]:
        positions[:,1] -= delta
    if keys_pressed[pygame.K_s]:
        positions[:,1] += delta
    if keys_pressed[pygame.K_a]:
        positions[:,0] += delta
    if keys_pressed[pygame.K_d]:
        positions[:,0] -= delta

    # For rotating it
    if keys_pressed[pygame.K_l]:
        anchor = positions[pinned_list, :].mean(axis=0)
        positions = ((positions - anchor) @ rot_mat(theta_delta).T) + anchor
    if keys_pressed[pygame.K_k]:
        anchor = positions[pinned_list, :].mean(axis=0)
        positions = ((positions - anchor) @ rot_mat(-theta_delta).T) + anchor



    # For saving UV map
    elif keys_pressed[pygame.K_s]:
        with open("face_uv.obj", 'w') as file:
            for p in positions:
                # Write each vertex with x, y from the list and z as 0
                file.write(f'v {p[0]} {p[1]} 0\n')


# Precompute drawing things that don't change (TODO: Ideally, I'd use something like Pyglet for fast draw)
# (a) Edge coloring
edge_cols = []
for i, (i1, i2) in enumerate(edges):
    percent = max(float(i / len(edges)), 0.4)
    edge_cols.append(percent * lerp(slash(percent, new_start=0.85, new_end=1.0), colors['darkpurple'], colors['lightyellow']))
    # if i1 not in special_verts and i2 not in special_verts:
    #     edge_cols.append(percent * lerp(percent, colors['lightblue'], colors['lightgreen']))
    # else:
    #     percent = min(float(i / len(edges)), 0.7)
    #     edge_cols.append(percent * colors['darkindigo'])

# Normal lerped coloring
# edge_cols = [lerp(i / len(edges), colors['black'], colors['white']) for i in range(len(edges))]
# # Override "important" edges with single color
# for i, (i1, i2) in enumerate(edges):
#     if i1 in special_verts or i2 in special_verts: edge_cols[i] = colors['darkindigo']


# (b) Quicker dot drawing ("draw()" is slow, so we draw it to a single surface, and blit that surface elsewhere,
#     which is faster apparently, because blitting is implemented in C. It's kinda like instanced rendering ig.)
def make_dot(color, radius=1):
    surf = pygame.Surface((2 * radius + 1, 2 * radius + 1), pygame.SRCALPHA)  # a tinyass surface for this dot
    pygame.draw.circle(surf, color, (radius, radius), radius)
    return surf

# Actually compute and stuff the two surfaces (white and red) for each of the two possible dots
dot_white = make_dot(colors['white'])
dot_red   = make_dot(colors['red'])

# Key-frame / Timing params
# Keyframe / timing params
t = 0.0
dt = 0.008    # i.e., 1 frame corresponds to +0.01 in parameter space = 0.01 * FPS = +0.6 per second (assuming 60 FPS)

# See google doc for better description of choreography.
keys = [0,      # Keyframe 0. After some delay, I wanna scramble it up, and then pick it apart (all while swirling)
        6.,     # Keyframe 1. Pick apart pinned vertices from rest of the network, run highlight
        9.,     # Keyframe 2. Expand them out, encircling the rest.
        12.,    # Keyframe 3. Simulate + zoom in / out!
        14.]    # Final Placeholder...


# Evolve step (main springy code)
def simulate():
    global positions, velocities, pinned_set, M, D, D_inv_op
    # Compute new velocities
    # 1. Reshape old velocity + position vectors
    v0 = velocities.reshape(-1, 1)
    p = positions.reshape(-1, 1)
    # 2. Compute new velocities
    v = v0 + (M @ (p + (h * v0)))
    v = v.reshape(len(positions), 2)
    # 3. Update velocities of only un-pinned vertices
    for row, vel in enumerate(v):
        if row not in pinned_set:
            velocities[row] = vel
    # 4. Update positions based on velocities
    positions += h * velocities


def main():
    global positions, velocities, t, dt, M, D, D_inv_op

    # Pre-gameloop stuff
    run = True
    clock = pygame.time.Clock()

    # Animation saving setup
    path_to_save = '/Users/adityaabhyankar/Desktop/Programming/UV_Unwrapping/output'
    if save_anim:
        for filename in os.listdir(path_to_save):
            # Check if the file name follows the required format
            b1 = filename.startswith("frame") and filename.endswith(".png")
            b2 = filename.startswith("output.mp4")
            if b1 or b2:
                os.remove(os.path.join(path_to_save, filename))
                print('Deleted frame ' + filename)

    # GLOBAL VARIABLES NEEDED FOR PER-KEYFRAME SPECIFIC CASE BY CASE BASIS ————————————————————————————————————
    # For keys (0) and (1), for the swirling.
    basepositions = copy.copy(positions)  # positions around which we swirl
    twirl_radii = np.array([4.0, 15.0])   # [global twirl radius, local twirl radius]
    twirl_radii0 = copy.copy(twirl_radii)  # just for lerping out of it

    # For key (0), for the scrambling up. (We'll LERP the basepositions between these.)
    bp0, bp1 = copy.copy(basepositions), np.zeros((len(mesh.vertices), 2))
    for i in range(bp1.shape[0]):
        x_max, y_max = width * 0.9, width * 0.9
        bp1[i] = np.random.rand(2) * [x_max, y_max] - [x_max / 2, y_max / 2]

    # For key (0), picking specific groups of vertices apart
    pickedlist1 = np.random.choice(positions.shape[0], 20, replace=False)
    pickedlist2 = np.random.choice(positions.shape[0], 15, replace=False)
    picked = set(np.concatenate([pickedlist1, pickedlist2])) # set, for fast look-up

    # For key (0), random permutation for swapping nodes around
    perm = np.random.permutation(pickedlist2)  # same length as pickedlist2

    # For key (0), the original basepositions of the subset of verts given by the permutation above
    pinned_bp0 = copy.copy(basepositions[pinned_list, :])
    pinned_bp1 = (pinned_bp0 / np.array([1.5, 3.0])) + np.array([0.0, 400.0])
    pinned_bp2 = []
    pinned_bp1_mean = np.mean(pinned_bp1, axis=0)
    for idx in pinned_list:
        angle = (np.pi * 2.0 * (main_boundary.index(idx) / float(len(main_boundary))))
        pinned_bp2.append(pinned_bp1_mean + (200.0 * np.array([np.cos(angle), np.sin(angle)])))
    pinned_bp2 = np.array(pinned_bp2)

    downwards = 200.0
    unpinned_bp0 = np.delete(basepositions, pinned_list, axis=0)
    unpinned_bp1 = unpinned_bp0 + np.array([0.0, -downwards])
    unpinned_mask = np.ones(basepositions.shape[0], dtype=bool)  # for MODIFYING the unpinned.
    unpinned_mask[pinned_list] = False

    # For key (2), the final locations of the pinned and unpinned verts (prior to beginning the simulation)
    pinned_pos_0 = copy.copy(pinned_bp2)
    pinned_pos_1 = (pinned_pos_0 - pinned_bp1_mean) * 1.7
    unpinned_pos_0 = copy.copy(unpinned_bp1)
    unpinned_pos_1 = (unpinned_pos_0 + np.array([0.0, downwards])) / 1.6








    # Game loop
    count = 0
    frame = -1
    while run:
        # Reset stuff
        window.fill((0, 0, 0))

        # MESH DRAWING ————————— We'll always need to draw the mesh, regardless of frame —————————————————————————
        # Draw edges
        P = A_many(positions)  # makes it much faster!!
        line = pygame.draw.line  # wow predefining this is actually slightly faster.
        for i, (i1, i2) in enumerate(edges):
            line(window, edge_cols[i], P[i1], P[i2], width=4 if i1 in special_verts and i2 in special_verts else 1)

        # Draw dots
        blit = window.blit  # wow predefining this is actually slightly faster.
        for i, pt in enumerate(P):
            spr = dot_white if i not in pinned_set else dot_red
            blit(spr, spr.get_rect(center=(int(pt[0]), int(pt[1]))))


        # ANIMATION ———————————————————————————————————————————————————————————————————————————————————————————————
        # Basic parameters
        prevframe = frame
        u, frame = squash(t)
        firstframe = frame != prevframe

        # Swirling motion needed for both of the first two keyframes.
        if frame in {0, 1}:
            # Add the circular twirls. We'll use "count" to parameterize the twirls,
            # so it's not dependent on the exact keyframe we're in.
            for i in range(positions.shape[0]):
                if i not in picked:
                    # A larger, global circular twirl.
                    radius = twirl_radii[0]
                    angle = ((count % 100) / 100. * 2.0 * np.pi)
                    positions[i] = basepositions[i] + (np.array([np.cos(angle), np.sin(angle)]) * radius)
                    # Add individual, circular twirls
                    radius = twirl_radii[1]
                    angle -= (deterministic_rand(i) * 2.0 * np.pi)
                    positions[i] += np.array([np.cos(angle), np.sin(angle)]) * radius

            # Keyframe 0. After some delay, I wanna scramble it up, and then pick it apart (all while swirling)
            if frame == 0:
                u = slash(u, new_start=0.1)  # short delay
                tau, seg = squash(u, [0.0,   # Seg 0: Scramble it up a bit.
                                      0.2,   # Seg 1: Pick first few vertices apart
                                      0.4,   # Seg 2: Drift back into the mess + pick next 6 vertices apart
                                      0.65,   # Seg 3: Flip some around so their edges cross
                                      0.85,   # Seg 4: Place them back into mess
                                      1.0])

                # Seg 0: Scramble it up.
                if seg == 0:
                    tau = ease_inout(tau)
                    basepositions = lerp(tau, bp0, bp1)

                # Seg 1: Pick first few vertices apart
                if seg == 1:
                    tau = ease_inout(tau)
                    for k, idx in enumerate(pickedlist1):
                        destination_y = height/2 * 0.8
                        if k % 2 == 0:
                            destination_y += 50.  # make some stick out more
                        if k > len(pickedlist1)/2:
                            destination_y *= -1  # opposite direction for some

                        destination_x = basepositions[idx][0]
                        positions[idx] = lerp(tau, basepositions[idx], np.array([destination_x, destination_y]))

                        # In the end, make final destination the new base point
                        if tau > 0.99:
                            basepositions[idx] = copy.copy(positions[idx])

                # Seg 2: Drift back into the mess + pick next 6 vertices apart
                elif seg == 2:
                    tau = slash(tau, new_start=0.3, new_end=1.0)  # a little pause
                    tau = ease_inout(tau)

                    # Put first set back (just toss em somewhere inside)
                    for idx in pickedlist1:
                        positions[idx] = lerp(tau, basepositions[idx], np.array([0., 0.]))

                        # In the end, make final destination the new base point
                        if tau > 0.98:
                            basepositions[idx] = copy.copy(positions[idx])

                    # Pull second set out
                    for k, idx in enumerate(pickedlist2):
                        destination_y = height/2 * 0.8
                        if k % 2 == 0:
                            destination_y += 50.  # make some stick out more
                        if k > len(pickedlist2)/2:
                            destination_y *= -1  # opposite direction for some

                        destination_x = basepositions[idx][0]
                        positions[idx] = lerp(tau, basepositions[idx], np.array([destination_x, destination_y]))

                        # In the end, make final destination the new base point
                        if tau > 0.99:
                            basepositions[idx] = copy.copy(positions[idx])

                # Seg 3: Drift back into the mess + pick next 6 vertices apart
                elif seg == 3:
                    tau = slash(tau, new_start=0.3, new_end=1.0)  # a little pause
                    tau = ease_inout(tau)

                    # Pick two from the second picked list and flip them around
                    idx1, idx2 = pickedlist2[0], pickedlist2[1]
                    if tau < 0.99:
                        destx = basepositions[perm, 0]  # destination x's
                        srcx = basepositions[pickedlist2, 0]  # current x's
                        positions[pickedlist2, 0] = lerp(tau, srcx, destx)  # smooth move

                    # In the end, make final destination the new base point
                    else:
                        basepositions[pickedlist2] = positions[pickedlist2].copy()

                # Seg 4: But them back into the mess
                elif seg == 4:
                    tau = slash(tau, new_start=0.3, new_end=1.0)  # a little pause
                    tau = ease_inout(tau)

                    # Put first set back (just toss em somewhere inside)
                    for idx in pickedlist2:
                        positions[idx] = lerp(tau, basepositions[idx], np.array([0., 0.]))

                        # In the end, make final destination the new base point
                        if tau > 0.99:
                            basepositions[idx] = copy.copy(positions[idx])

            # Keyframe 1. Pick apart pinned vertices from rest of the network, run highlight
            if frame == 1:
                u = slash(u, new_start=0.1)  # short delay
                tau, seg = squash(u, [0.0,   # Seg 0: Pick apart pinned vertices
                                      0.33,  # Seg 1: Rearrange them into a circle
                                      0.66,  # Seg 2: Run the highlight.
                                      1.0])

                # Seg 0: Pick apart pinned vertices
                if seg == 0:
                    tau = ease_inout(tau)
                    # Move the pinned vertices up above
                    basepositions[pinned_list, :] = lerp(tau, pinned_bp0, pinned_bp1)
                    # Move the unpinned vertices downwards a bit
                    basepositions[unpinned_mask] = lerp(tau, unpinned_bp0, unpinned_bp1)
                    # Stop the swirling
                    twirl_radii = lerp(tau, twirl_radii0, np.array([0.0, 0.0]))

                # Seg 1: Rearrange them into a circle
                if seg == 1:
                    tau = ease_inout(tau)
                    basepositions[pinned_list, :] = lerp(tau, pinned_bp1, pinned_bp2)

                # Seg 2. Run the highlight.
                if seg == 2:
                    tau = ease_inout(tau)
                    start_idx = int(slash(tau, new_start=0.9) * len(pinned_list))
                    end_idx = int(tau * len(pinned_list))
                    line_pts = A_many(basepositions[main_boundary[start_idx:end_idx]])

                    if len(line_pts) > 1: pygame.draw.lines(window, (255, 255, 255), False, line_pts, width=4)

        # (NO MORE SWIRLING!)
        # Keyframe 2. Expand them out, encircling the rest.
        if frame == 2:
            u = slash(u, new_start=0.1)  # short delay
            u = ease_inout(u)
            positions[pinned_list, :] = lerp(u, pinned_pos_0, pinned_pos_1)
            positions[unpinned_mask] = lerp(u, unpinned_pos_0, unpinned_pos_1)

        # Keyframe 3. Simulate!
        if frame == 3:
            # We'll use BASE POSITIONS as the main simulation array. That way "positions" can be
            # used for zooming in / out. Also let's make those basepositions into np array.
            if firstframe:
                basepositions = np.array(copy.copy(positions))

                # Setup the solver
                # (1)
                D = sp.sparse.identity(CTC.shape[0], format='csc') + (h * h * stiff) * CTC
                D_inv_op = sp.sparse.linalg.factorized(D)  # note: this is an OPERATOR, not an explicit matrix!
                # (2) Compute sparse solver matrix as a linear operator (also not an explicit matrix)
                M = sp.sparse.linalg.LinearOperator((2 * len(positions), 2 * len(positions)), matvec=lambda x: -h * stiff * D_inv_op(CTC @ x))

            # A little pause
            u = slash(u, new_start=0.2)
            if u > 0.0:
                # Start simulation:
                simulate()

                # Zoom in/out etc keyframes
                u = slash(u, new_start=0.3)  # another little pause, before doing these zoom-ins
                tau, seg = squash(u, [0.0,   # Seg 0: TODO
                                      1.0])










        # Handle keys
        keys_pressed = pygame.key.get_pressed()
        handle_keys(keys_pressed)

        # End run (1. Tick clock, 2. Save the frame, and 3. Detect if window closed)
        pygame.display.update()
        clock.tick(60)  # FPS
        count += 1
        t += dt
        if save_anim:
            pygame.image.save(window, path_to_save + '/frame' + str(count) + '.png')
            print('Saved frame ' + str(count))
        # elif frame < len(keys)-1:
        #     print(clock.get_fps())
        #     # print('Frame:', frame, ', u:', u)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()

    # POST Gameloop stuff
    # Use ffmpeg to combine the PNG images into a video
    if save_anim:
        input_files = path_to_save + '/frame%d.png'
        output_file = path_to_save + '/output.mp4'
        ffmpeg_path = "/opt/homebrew/bin/ffmpeg"
        os.system(
            f'{ffmpeg_path} -r 60 -i {input_files} -vcodec libx264 -crf 25 -pix_fmt yuv420p -vf "eq=brightness=0.00:saturation=1.3" {output_file} > /dev/null 2>&1')
        print('Saved video to ' + output_file)


if __name__ == '__main__':
    main()