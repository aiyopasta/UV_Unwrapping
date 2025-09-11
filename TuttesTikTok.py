# This code resulted in an Instagram reel that got 3.5 MILLION views!! Congrats!!

import copy
import numpy as np
import pygame
import os
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)

# Save the animation? TODO: Make sure you're saving to correct destination!!
save_anim = False

# Pygame + gameloop setup
width = 540
height = 960
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Tutte's Embedding Animations")
pygame.init()


# Drawing Coordinate Shift Functions
def A(val):
    return np.array([val[0] + width / 2, -val[1] + height / 2])


def A_inv(val):
    global width, height
    return np.array([val[0] - width / 2, -(val[1] - height / 2)])


def A_many(vals):
    return [A(v) for v in vals]   # Galaxy brain function


# Keyframe / timing params
FPS = 60
t = 0.0
dt = 0.01    # i.e., 1 frame corresponds to +0.01 in parameter space = 0.01 * FPS = +0.6 per second (assuming 60 FPS)

keys = [0,      # Keyframe 0. Dancing messy network
        3.,     # Keyframe 1. Picking it apart.
        6.,     # Keyframe 2. Pick a triangle.
        10.,     # Keyframe 3. Tuttes for triangle boundary. TODO: MAKE LONGER IN ACTUAL RENDER
        11.,     # Keyframe 4. Pick the mouth hole.
        13.,    # Keyframe 5. Tuttes for mouth hole boundary  TODO MAKE LONGER IN ACTUAL RENDER
        14.,    # Keyframe 6. Pick the true boundary hole.
        16.,    # Keyframe 7. Tuttes for true boundary hole  TODO MAKE LONGER IN ACTUAL RENDER
        20.]

# TODO (for now, to inspect the triangle embedding)
for k in range(4, len(keys)):
    keys[k] += 7.
for k in range(6, len(keys)):
    keys[k] += 3.


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


# Squeeze actual interpolation to be within [new_start, new_end], and make it 0 and 1 outside this range
def slash(t_, new_start=0.0, new_end=0.5):
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


# Mesh information (input)
filename = 'THE_ACTUAL_FACE_TO_USE.obj'  # use "THE ACTUAL FACE TO USE"

# FOR "THE ACTUAL FACE TO USE"
# RIGHT EYE [112, 367, 125, 376, 119, 364, 123, 365, 114, 369, 122, 363, 117, 375, 126, 370]
# RANDOM TRIANGLE [67, 482, 207]

# FOR FACE 7
# Random loop [21, 151, 48, 237, 64, 270, 101, 262, 104, 258, 478, 177, 464, 228, 472, 252, 608, 657, 638, 648, 636, 646, 632, 643, 631, 675, 615, 41, 163, 42, 205]
# FROM MOUTH [629, 641, 630, 645, 633, 644, 654, 655, 653, 637, 647, 639, 649, 640, 651, 634, 650, 635, 652, 638, 648, 636, 646, 632, 643, 631, 642]
# FROM EYE [112, 367, 125, 376, 119, 364, 123, 365, 114, 369, 122, 363, 117, 375, 126, 370]
# DUMB TRIANGLE [67, 482, 207]
# FROM TRUE BOUNDARY [128, 418, 142, 417, 143, 412, 149, 409, 134, 425, 133, 422, 137, 407, 147, 414, 146, 410, 145, 415, 144, 408, 141, 419, 140, 416, 136, 423, 135, 424, 132, 426, 131, 421, 138, 420, 139, 413, 148, 411, 130, 598, 427, 129, 428]

# FOR FACE 6, 5, 4 ...
#[128, 418, 142, 417, 143, 412, 149, 409, 134, 425, 133, 422, 137, 407, 147, 414, 146, 410, 145, 415, 144, 408, 141, 419, 140, 416, 136, 423, 135, 424, 132, 426, 131, 421, 138, 420, 139, 413, 148, 411, 130, 598, 427, 129, 428]
#[143, 465, 157, 464, 158, 459, 164, 456, 149, 472, 148, 469, 152, 454, 162, 461, 161, 457, 160, 462, 159, 455, 156, 466, 155, 463, 151, 470, 150, 471, 147, 473, 146, 468, 153, 467, 154, 460, 163, 458, 145, 474, 144, 475]
#[163, 525, 177, 524, 178, 519, 184, 516, 169, 532, 168, 529, 172, 514, 182, 521, 181, 517, 180, 522, 179, 515, 176, 526, 175, 523, 171, 530, 170, 531, 167, 533, 166, 528, 173, 527, 174, 520, 183, 518, 165, 534, 164, 535]
#[163, 525, 177, 524, 178, 519, 184, 516, 169, 532, 168, 529, 172, 514, 182, 521, 181, 517, 180, 522, 179, 515, 176, 526, 175, 523, 171, 530, 170, 531, 167, 533, 166, 528, 173, 527, 174, 520, 183, 518, 165, 534, 164, 535]
#[163, 177, 178, 184, 169, 168, 172, 182, 181, 180, 179, 176, 175, 171, 170, 167, 166, 173, 174, 183, 165, 164]
 #[168, 166, 180, 171, 172, 170, 178, 164, 169, 185, 176, 182, 165, 179, 174, 175, 173, 181, 167, 183, 177, 184]

# Load OBJ file + fill edges list
positions = []
faces = []
edges = set()
delimiter = '/'
with open(filename) as f:
    for line in f:
        kind = line[0]
        if kind == 'v':
            pass
            positions.append([float(l) for l in line.rsplit()[1:]])
        elif kind == 'f':
            indices = None
            if delimiter in line:
                indices = [int(l[:l.find(delimiter)]) - 1 for l in line.rsplit()[1:]]
            else:
                indices = [int(l) - 1 for l in line.rsplit()[1:]]

            faces.append(indices)

            # Add edges for each face
            num_vertices = len(indices)
            for i in range(num_vertices):
                # Create an edge (vi, vj), ensuring vi < vj for uniqueness
                vi = indices[i]
                vj = indices[(i + 1) % num_vertices]  # Wrap around to the first vertex
                edge = (min(vi, vj), max(vi, vj))
                edges.add(edge)

edges = list(edges)

# Set positions to 2D values, and as a numpy array
positions = np.array([[0.0, 0.0] for _ in range(len(positions))])

# Initialize velocities numpy array
velocities = np.zeros((len(positions), 2))


# Additional vars
colors = {
    'white': np.array([255., 255., 255.]),
    'black': np.array([0., 0., 0.]),
    'red': np.array([255, 66, 48]),
    'blue': np.array([30, 5, 252]),
    'fullred': np.array([255, 0, 0]),
    'fullblue': np.array([0, 0, 255]),
    'START': np.array([255, 255, 255]),
    'orange': np.array([255, 165, 0])
}


# Keyhandling (for zooming in and out)
def handle_keys(keys_pressed):
    global positions

    if keys_pressed[pygame.K_p]:
        positions = [pos * 1.5 for pos in positions]
    elif keys_pressed[pygame.K_o]:
        positions = [pos / 1.5 for pos in positions]


def main():
    global t, dt, keys, FPS, save_anim, colors, edges, positions, velocities

    # Pre-animation setup
    clock = pygame.time.Clock()
    run = True

    # Animation saving setup
    path_to_save = '/Users/adityaabhyankar/Desktop/Programming/LaplaceMania/for_animation/output'
    if save_anim:
        for filename in os.listdir(path_to_save):
            # Check if the file name follows the required format
            b1 = filename.startswith("frame") and filename.endswith(".png")
            b2 = filename.startswith("output.mp4")
            if b1 or b2:
                os.remove(os.path.join(path_to_save, filename))
                print('Deleted frame ' + filename)

    # For frames 0, 1 — Store randomize base positional values
    np.random.seed(40)
    basepositions = copy.copy(positions)
    for i in range(len(positions)):
        x_max, y_max = width * 0.9, width * 0.9
        basepositions[i] = np.random.rand(2) * [x_max, y_max] - [x_max / 2, y_max / 2]

    # For frame 1 — Store initial positions of a handful of vertices (to be picked out)
    pickedlist1 = np.random.choice(len(positions), 6, replace=False)
    pickedlist2 = np.random.choice(len(positions), 6, replace=False)

    # For frame 2 — Store list of candidate triangles to visit
    candtris = [[20, 495, 334],
                [297, 366, 495],
                [296, 366, 115],
                [110, 496, 300],
                [29, 496, 341],
                [301, 373, 496],
                [300, 373, 114],
                [67, 497, 303],
                [235, 282, 497],
                [282, 316, 497],
                [303, 316, 89],
                [175, 302, 75],
                [175, 303, 498],
                [498, 89, 307],
                [67, 482, 207]]    # <--- That's the chosen triangle we're gonna roll with

    # For frame 2 — Initial twirling radii
    radii = np.array([3., 10.])

    # For frame 2 — Transparency channel for fading out triangle
    transparent_surface = pygame.Surface((width, height), pygame.SRCALPHA)

    # For frame 2 — For changing thickness of lines.
    thickness = 1

    # For frame 4 — Indices for the mouth hole
    hole_verts = [629, 641, 630, 645, 633, 644, 654, 655, 653, 637, 647, 639, 649, 640, 651, 634, 650, 635, 652, 638, 648, 636, 646, 632, 643, 631, 642] #[112, 367, 125, 376, 119, 364, 123, 365, 114, 369, 122, 363, 117, 375, 126, 370]

    # For frame 5 — Indices for the true boundary hole
    boundary_hole_verts = [128, 418, 142, 417, 143, 412, 149, 409, 134, 425, 133, 422, 137, 407, 147, 414, 146, 410, 145, 415, 144, 408, 141, 419, 140, 416, 136, 423, 135, 424, 132, 426, 131, 421, 138, 420, 139, 413, 148, 411, 130, 598, 427, 129, 428]

    # Initialize 2|E| x 2|V| sized connections matrix, and blank solver matrix M
    C = np.zeros((2 * len(edges), 2 * len(basepositions)))
    for row, e in enumerate(edges):
        i1, i2 = e
        # x-dimension constraint
        C[2 * row][2 * i1] = 1
        C[2 * row][2 * i2] = -1
        # y-dimension constraint
        C[(2 * row) + 1][(2 * i1) + 1] = 1
        C[(2 * row) + 1][(2 * i2) + 1] = -1
    M = None

    # Game loop
    count = 0
    frame = -1

    while run:
        # Reset stuff
        window.fill((0, 0, 0))

        # Draw edges
        for i, e in enumerate(edges):
            v1, v2 = positions[e[0]], positions[e[1]]
            u = i / len(edges)
            if (e[0] in hole_verts) and (e[1] in hole_verts):
                pygame.draw.line(window, colors['orange'], A(v1), A(v2), width=4)
            else:
                pygame.draw.line(window, lerp(u, colors['blue'], colors['red']), A(v1), A(v2), width=thickness)

        # Draw nodes
        for i, p in enumerate(positions):
            pygame.draw.circle(window, colors['white'], A(p), 2, width=0)

        # Animation!
        prevframe = frame
        u, frame = squash(t)
        firstframe = frame != prevframe
        # Keyframe 0 + 1 + 2 — Dancing messy network, picking it apart, and picking a triangle.
        np.random.seed(42)
        if frame in {0, 1, 2}:
            # Add circular twirls
            for i in range(len(positions)):
                # Set flags for which ones not to twirl TODO
                if (i not in pickedlist1) and (i not in pickedlist2):
                    # Add global circular twirl
                    radius = radii[0]
                    angle = ((count % 100) / 100. * 2.0 * np.pi)
                    positions[i] = basepositions[i] + (np.array([np.cos(angle), np.sin(angle)]) * radius)
                    # Add small circular twirls
                    radius = radii[1]
                    angle -= (np.random.rand() * 2.0 * np.pi)
                    positions[i] += np.array([np.cos(angle), np.sin(angle)]) * radius

            # Keyframe 1 Stuff
            if frame == 1:
                tau, seg = squash(u, [0.0,   # Seg 0: Pick first 6 vertices apart
                                      0.3,   # Seg 1: Drift back into the mess + pick next 6 vertices apart
                                      0.6,   # Seg 2: Flip some around so their edges cross
                                      0.8,   # Seg 3: Place them back into mess
                                      1.0])

                # Pick first 6 vertices apart
                if seg == 0:
                    tau = ease_out(tau)
                    for k, idx in enumerate(pickedlist1):
                        destination_y = height/2 * 0.8
                        if k % 2 == 0:
                            destination_y += 50.  # make some stick out more
                        if k > 2:
                            destination_y *= -1  # opposite direction for some

                        destination_x = basepositions[idx][0]
                        positions[idx] = lerp(tau, basepositions[idx], np.array([destination_x, destination_y]))

                        # In the end, make final destination the new base point
                        if tau > 0.99:
                            basepositions[idx] = copy.copy(positions[idx])

                # Drift back into the mess + pick next 6 vertices apart
                elif seg == 1:
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
                        if k > 2:
                            destination_y *= -1  # opposite direction for some

                        destination_x = basepositions[idx][0]
                        positions[idx] = lerp(tau, basepositions[idx], np.array([destination_x, destination_y]))

                        # In the end, make final destination the new base point
                        if tau > 0.99:
                            basepositions[idx] = copy.copy(positions[idx])

                # Drift back into the mess + pick next 6 vertices apart
                elif seg == 2:
                    tau = slash(tau, new_start=0.3, new_end=1.0)  # a little pause
                    tau = ease_inout(tau)

                    # Pick two from the second picked list and flip them around
                    idx1, idx2 = pickedlist2[0], pickedlist2[1]
                    if tau < 0.99:
                        destx1, destx2 = basepositions[idx2][0], basepositions[idx1][0]
                        positions[idx1][0] = lerp(tau, basepositions[idx1][0], destx1)
                        positions[idx2][0] = lerp(tau, basepositions[idx2][0], destx2)

                    # In the end, make final destination the new base point
                    else:
                        basepositions[idx1] = copy.copy(positions[idx1])
                        basepositions[idx2] = copy.copy(positions[idx2])

                # Put them all back into mess
                elif seg == 3:
                    tau = slash(tau, new_start=0.3, new_end=1.0)  # a little pause
                    tau = ease_inout(tau)

                    # Put first set back (just toss em somewhere inside)
                    for idx in pickedlist2:
                        positions[idx] = lerp(tau, basepositions[idx], np.array([0., 0.]))

                        # In the end, make final destination the new base point
                        if tau > 0.99:
                            basepositions[idx] = copy.copy(positions[idx])

            # Keyframe 2 Stuff
            if frame == 2:
                tau, seg = squash(u, [0.0,  # Seg 0: Flip through many candidate triangles, ending up on one
                                      0.7,  # Seg 1: Expand out that triangle (may need to zoom out to fit everything)
                                      1.0])

                # Flip through many candidate triangles, ending up on one
                if seg == 0:
                    tau = ease_inout(tau)
                    sigma, tri_idx = squash2(tau, n_intervals=15)
                    # Draw the triangle with thick lines to point it out
                    tri = candtris[tri_idx]
                    pygame.draw.lines(window, colors['white'] * sigma, True, A_many([positions[tri[0]], positions[tri[1]], positions[tri[2]]]), width=8)

                    # LERP the twirl radii to 0, to bring the points back to their basepoints
                    radii = lerp(tau, np.array([3., 10.]), np.array([0., 0.]))


                # Expand out that triangle (may need to zoom out to fit everything, or squeeze everything else in haha)
                if seg == 1:
                    tau = ease_inout(tau)
                    tri = candtris[14]

                    # LERP triangle to boundary
                    angle_offset = np.radians(-120.) - np.radians(30.)
                    for i, vert in enumerate(tri):
                        angle = (np.pi * 2.0 * (i / 3.)) + angle_offset
                        destpos = np.array([np.cos(angle), np.sin(angle)]) * min(width, height) / 2
                        destpos += np.array([5., 0.])
                        positions[tri[i]] = lerp(tau, basepositions[tri[i]], destpos)

                    # LERP rest of the nodes to be a bit smaller (to fit inside triangle)
                    factor = 0.4
                    for idx in range(len(positions)):
                        if idx not in tri:
                            positions[idx] = basepositions[idx] * lerp(tau, 1.0, factor)

                    # Draw triangle, fade out white highlight
                    pygame.draw.lines(transparent_surface, np.array([*colors['white'], 255. * (1 - tau)]), True,
                                      A_many([positions[tri[0]], positions[tri[1]], positions[tri[2]]]), width=8)

                    window.blit(transparent_surface, (0, 0))

        # Keyframe 3 — Tuttes for triangle boundary
        elif frame == 3:
            # We'll use BASE POSITIONS as the main simulation array. That way "positions" can be
            # used for zooming in / out. Also let's make those basepositions into np array.
            if firstframe:
                basepositions = np.array(copy.copy(positions))

                # Sneakily increase thickness of lines
                thickness += 1

            # A little pause
            u = slash(u, new_start=0.3)
            u = ease_inout(u)

            # Spring + simulation parameters
            stiff = 800  # 5000
            h = 0.01  # 0.01
            pinned = [67, 482, 207]

            if firstframe:
                CTC = C.T @ C
                M = -h * stiff * np.linalg.inv(np.eye(2 * len(basepositions)) + (np.power(h, 2.) * stiff * CTC)) @ CTC

            # SIMULATE
            # Compute new velocities
            # 1. Reshape old velocity + position vectors
            v0 = velocities.reshape(-1, 1)
            p = basepositions.reshape(-1, 1)
            # 2. Compute update matrix
            # CTC = C.T @ C
            # M = -h * stiff * np.linalg.inv(np.eye(2 * len(basepositions)) + (np.power(h, 2.) * stiff * CTC)) @ CTC
            # 3. Compute new velocities
            v = v0 + (M @ (p + (h * v0)))
            v = v.reshape(len(basepositions), 2)
            # 4. Update velocities of only un-pinned vertices
            for row, vel in enumerate(v):
                if row not in pinned:
                    velocities[row] = vel
            # 5. Update basepositions based on velocities
            basepositions += h * velocities

            # Apply zoom, and update positions themselves
            max_zoom = 3.0
            positions = np.array([basepos * lerp(u, 1.0, max_zoom) for basepos in basepositions])

        # Keyframe 4 — Pick mouth hole
        elif frame == 4:
            # Base positions will be the positions from the final instant of the previous frame
            if firstframe:
                basepositions = np.array(copy.copy(positions))

            tau, seg = squash(u, [0.0,   # Seg 0: Highlight hole
                                  0.5,   # Seg 1: Expand it out (may need to zoom out to fit everything)
                                  1.0])

            # Highlight the hole
            if seg == 0:
                tau = ease_inout(tau)
                # Iterate through the hole's boundary verts, drawing in the edges one by one
                sigma, edge_num = squash2(tau, n_intervals=len(hole_verts)-1)  # edge_num is the name of the SECTION, i.e. it gives the # of the edge being drawn
                for i in range(edge_num+1):
                    startpos = positions[hole_verts[i]]
                    endpos = positions[hole_verts[(i+1) % len(hole_verts)]]
                    # LERP-in the current edge
                    if i == edge_num:
                        midpos = lerp(sigma, startpos, endpos)
                        pygame.draw.line(window, colors['white'], A(startpos), A(midpos), width=8)

                    # Just straight draw in all the previously drawn ones
                    pygame.draw.line(window, colors['white'], A(startpos), A(endpos), width=8)

            # Expand it out (may need to zoom out to fit everything)
            if seg == 1:
                tau = ease_inout(tau)
                # LERP hole to boundary
                angle_offset = np.radians(-120.)
                for i, vert in enumerate(hole_verts):
                    angle = (np.pi * 2.0 * (i / len(hole_verts))) + angle_offset
                    destpos = np.array([np.cos(angle), np.sin(angle)]) * min(width, height) / 2 * 0.9
                    destpos -= np.array([0., 0.])
                    positions[hole_verts[i]] = lerp(tau, basepositions[hole_verts[i]], destpos)

                # LERP rest of the nodes to be a bit smaller (to fit inside the circle)
                factor = 0.25
                for idx in range(len(positions)):
                    if idx not in hole_verts:
                        positions[idx] = basepositions[idx] * lerp(tau, 1.0, factor)

                # Draw triangle, fade out white highlight
                pygame.draw.lines(transparent_surface, np.array([*colors['white'], 255. * (1 - tau)]), True,
                                  A_many([positions[hole_idx] for hole_idx in hole_verts]), width=8)

                window.blit(transparent_surface, (0, 0))

        # Keyframe 5 — Tuttes for mouth hole boundary
        elif frame == 5:
            # We'll use BASE POSITIONS as the main simulation array. That way "positions" can be
            # used for zooming in / out. Also let's make those basepositions into np array.
            # Reset velocities to 0 too.
            if firstframe:
                basepositions = np.array(copy.copy(positions))
                velocities = np.zeros((len(positions), 2))

            # A little pause
            u = slash(u, new_start=0.1)
            u = ease_inout(u)

            # Spring + simulation parameters
            stiff = 2000  # 5000
            h = 0.01  # 0.01
            pinned = hole_verts

            if firstframe:
                CTC = C.T @ C
                M = -h * stiff * np.linalg.inv(np.eye(2 * len(basepositions)) + (np.power(h, 2.) * stiff * CTC)) @ CTC

            # SIMULATE
            # Compute new velocities
            # 1. Reshape old velocity + position vectors
            v0 = velocities.reshape(-1, 1)
            p = basepositions.reshape(-1, 1)
            # 2. Compute update matrix
            # CTC = C.T @ C
            # M = -h * stiff * np.linalg.inv(np.eye(2 * len(basepositions)) + (np.power(h, 2.) * stiff * CTC)) @ CTC
            # 3. Compute new velocities
            v = v0 + (M @ (p + (h * v0)))
            v = v.reshape(len(basepositions), 2)
            # 4. Update velocities of only un-pinned vertices
            for row, vel in enumerate(v):
                if row not in pinned:
                    velocities[row] = vel
            # 5. Update basepositions based on velocities
            basepositions += h * velocities

            # 6. Update positions themselves
            positions = np.array([basepos for basepos in basepositions])

        # Keyframe 6 — Pick the true boundary hole.
        elif frame == 6:
            # Base positions will be the positions from the final instant of the previous frame
            if firstframe:
                basepositions = np.array(copy.copy(positions))

            tau, seg = squash(u, [0.0,  # Seg 0: Highlight hole
                                  0.5,  # Seg 1: Expand it out (may need to zoom out to fit everything)
                                  1.0])

            # Highlight the hole
            if seg == 0:
                tau = ease_inout(tau)
                # Iterate through the hole's boundary verts, drawing in the edges one by one
                sigma, edge_num = squash2(tau, n_intervals=len(boundary_hole_verts) - 1)  # edge_num is the name of the SECTION, i.e. it gives the # of the edge being drawn
                for i in range(edge_num + 1):
                    startpos = positions[boundary_hole_verts[i]]
                    endpos = positions[boundary_hole_verts[(i + 1) % len(boundary_hole_verts)]]
                    # LERP-in the current edge
                    if i == edge_num:
                        midpos = lerp(sigma, startpos, endpos)
                        pygame.draw.line(window, colors['white'], A(startpos), A(midpos), width=8)

                    # Just straight draw in all the previously drawn ones
                    pygame.draw.line(window, colors['white'], A(startpos), A(endpos), width=8)

            # Expand it out (may need to zoom out to fit everything)
            if seg == 1:
                tau = ease_inout(tau)
                # LERP hole to boundary
                angle_offset = np.radians(-120.)
                for i, vert in enumerate(boundary_hole_verts):
                    angle = (np.pi * 2.0 * (i / len(boundary_hole_verts))) + angle_offset
                    destpos = np.array([np.cos(angle), np.sin(angle)]) * min(width, height) / 2 * 0.9
                    destpos -= np.array([0., 0.])
                    positions[boundary_hole_verts[i]] = lerp(tau, basepositions[boundary_hole_verts[i]], destpos)

                # LERP rest of the nodes to be a bit smaller (to fit inside the circle)
                factor = 0.7
                for idx in range(len(positions)):
                    if idx not in boundary_hole_verts:
                        positions[idx] = basepositions[idx] * lerp(tau, 1.0, factor)

                # Draw triangle, fade out white highlight
                pygame.draw.lines(transparent_surface, np.array([*colors['white'], 255. * (1 - tau)]), True,
                                  A_many([positions[hole_idx] for hole_idx in boundary_hole_verts]), width=8)

                window.blit(transparent_surface, (0, 0))

        # Keyframe 7 — Tuttes for true hole boundary
        elif frame == 7:
            # We'll use BASE POSITIONS as the main simulation array. That way "positions" can be
            # used for zooming in / out. Also let's make those basepositions into np array.
            # Reset velocities to 0 too.
            if firstframe:
                basepositions = np.array(copy.copy(positions))
                velocities = np.zeros((len(positions), 2))

            # A little pause
            u = slash(u, new_start=0.1)
            u = ease_inout(u)

            # Spring + simulation parameters
            stiff = 4000  # 5000
            h = 0.01  # 0.01
            pinned = boundary_hole_verts

            if firstframe:
                CTC = C.T @ C
                M = -h * stiff * np.linalg.inv(np.eye(2 * len(basepositions)) + (np.power(h, 2.) * stiff * CTC)) @ CTC

            # SIMULATE
            # Compute new velocities
            # 1. Reshape old velocity + position vectors
            v0 = velocities.reshape(-1, 1)
            p = basepositions.reshape(-1, 1)
            # 2. Compute update matrix
            # CTC = C.T @ C
            # M = -h * stiff * np.linalg.inv(np.eye(2 * len(basepositions)) + (np.power(h, 2.) * stiff * CTC)) @ CTC
            # 3. Compute new velocities
            v = v0 + (M @ (p + (h * v0)))
            v = v.reshape(len(basepositions), 2)
            # 4. Update velocities of only un-pinned vertices
            for row, vel in enumerate(v):
                if row not in pinned:
                    velocities[row] = vel
            # 5. Update basepositions based on velocities
            basepositions += h * velocities

            # 6. Update positions themselves
            positions = np.array([basepos for basepos in basepositions])


        else:
            print('done')



        # We handle keys pressed inside the gameloop in PyGame
        keys_pressed = pygame.key.get_pressed()
        handle_keys(keys_pressed)


        # End run (1. Tick clock, 2. Save the frame, and 3. Detect if window closed)
        pygame.display.flip()
        transparent_surface.fill((0, 0, 0, 0))
        t += dt
        # clock.tick(FPS)
        count += 1
        if save_anim:
            pygame.image.save(window, path_to_save+'/frame'+str(count)+'.png')
            print('Saved frame '+str(count))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()


    # Post game-loop stuff
    # Use ffmpeg to combine the PNG images into a video
    if save_anim:
        input_files = path_to_save + '/frame%d.png'
        output_file = path_to_save + '/output.mp4'
        ffmpeg_path = "/opt/homebrew/bin/ffmpeg"
        os.system(f'{ffmpeg_path} -r 60 -i {input_files} -vcodec libx264 -crf 25 -pix_fmt yuv420p -vf "eq=brightness=0.00:saturation=1.3" {output_file} > /dev/null 2>&1')
        print('Saved video to ' + output_file)

        # Save the final positions into new obj file, to import into Blender
        with open("face_uv.obj", 'w') as file:
            for p in positions:
                # Write each vertex with x, y from the list and z as 0
                file.write(f'v {p[0]} {p[1]} 0\n')

        print('Saved final positions to face_uv.obj')


if __name__ == "__main__":
    main()
