# Interesting find: As I increase the time-step, the drawing gets more and more concentrated towards the center.

import numpy as np
import pygame
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)

# Pygame + gameloop setup
width = 800
height = 800
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Tutte's Embedding")
pygame.init()


# Drawing Coordinate Shift Functions
def A(val):
    return np.array([val[0] + width / 2, -val[1] + height / 2])


def A_inv(val):
    global width, height
    return np.array([val[0] - width / 2, -(val[1] - height / 2)])


def A_many(vals):
    return [A(v) for v in vals]   # Galaxy brain function


# Keyhandling (modify + redraw call)
def handle_keys(keys_pressed):
    pass


# Mesh information (input)
filename = 'lowpolyface4.obj'
pinned = [163, 525, 177, 524, 178, 519, 184, 516, 169, 532, 168, 529, 172, 514, 182, 521, 181, 517, 180, 522, 179, 515, 176, 526, 175, 523, 171, 530, 170, 531, 167, 533, 166, 528, 173, 527, 174, 520, 183, 518, 165, 534, 164, 535]
#[163, 525, 177, 524, 178, 519, 184, 516, 169, 532, 168, 529, 172, 514, 182, 521, 181, 517, 180, 522, 179, 515, 176, 526, 175, 523, 171, 530, 170, 531, 167, 533, 166, 528, 173, 527, 174, 520, 183, 518, 165, 534, 164, 535]
#[163, 177, 178, 184, 169, 168, 172, 182, 181, 180, 179, 176, 175, 171, 170, 167, 166, 173, 174, 183, 165, 164]
 #[168, 166, 180, 171, 172, 170, 178, 164, 169, 185, 176, 182, 165, 179, 174, 175, 173, 181, 167, 183, 177, 184]
stiff = 3000
h = 0.02
# drag = 0.00009

# Load OBJ file + fill adjacency list
# 1. Load the raw vertices, edges, and faces list.
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

# 2. Override positions with randomized values if unpinned, else circular values.
np.random.seed(42)
angle_offset = np.radians(-120.)
for i in range(len(positions)):
    if i not in pinned:
        x_max, y_max = width * 0.5, height * 0.5
        positions[i] = np.random.rand(2) * [x_max, y_max] - [x_max / 2, y_max / 2]
        maxvel = 2
    else:
        angle = (np.pi * 2.0 * (pinned.index(i) / float(len(pinned)))) + angle_offset
        positions[i] = np.array([np.cos(angle), np.sin(angle)]) * min(width, height) / 2 * 0.9

# 3. Initialize velocities to be 0 and make positions into np array.
positions = np.array(positions)
velocities = np.zeros((len(positions), 2))

# 4. Initialize 2|E| x 2|V| sized connections matrix
C = np.zeros((2 * len(edges), 2 * len(positions)))
for row, e in enumerate(edges):
    i1, i2 = e
    # x-dimension constraint
    C[2 * row][2 * i1] = 1
    C[2 * row][2 * i2] = -1
    # y-dimension constraint
    C[(2 * row) + 1][(2 * i1) + 1] = 1
    C[(2 * row) + 1][(2 * i2) + 1] = -1


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


def main():
    global positions, velocities

    # Pre-gameloop stuff
    run = True
    clock = pygame.time.Clock()

    # Game loop
    count = 0
    while run:
        # Reset / increment stuff
        count += 1
        window.fill((0, 0, 0))
        # Compute new velocities
        # 1. Reshape old velocity + position vectors
        v0 = velocities.reshape(-1, 1)
        p = positions.reshape(-1, 1)
        # 2. Compute update matrix
        CTC = C.T @ C
        M = -h * stiff * np.linalg.inv(np.eye(2 * len(positions)) + (np.power(h, 2.) * stiff * CTC)) @ CTC
        # 3. Compute new velocities
        v = v0 + (M @ (p + (h * v0)))
        v = v.reshape(len(positions), 2)
        # 4. Update velocities of only un-pinned vertices
        for row, vel in enumerate(v):
            if row not in pinned:
                velocities[row] = vel
        # 5. Update positions based on velocities
        positions += h * velocities



        # Draw edges
        for e in edges:
            v1, v2 = positions[e[0]], positions[e[1]]
            pygame.draw.line(window, colors['blue'], A(v1), A(v2), width=1)

        # Draw nodes
        for i, p in enumerate(positions):
            pygame.draw.circle(window, colors['white'] if i not in pinned else colors['red'], A(p), 3, width=0)



        # End run (1. Tick clock, 2. Save the frame, and 3. Detect if window closed)
        pygame.display.update()
        clock.tick(60)  # FPS
        count += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()


if __name__ == '__main__':
    main()



# # NOTES: Shows some signs of working. It briefly seems to work, but quickly gets unstable. Making the
# #        springs stiffer should probably solve the problem, but explicit methods aren't enough to
# #        support stiff stuff.
#
# import numpy as np
# import pygame
# np.set_printoptions(suppress=True)
#
# # Pygame + gameloop setup
# width = 800
# height = 800
# window = pygame.display.set_mode((width, height))
# pygame.display.set_caption("1D Lorentz Transformation")
# pygame.init()
#
#
# # Drawing Coordinate Shift Functions
# def A(val):
#     return np.array([val[0] + width / 2, -val[1] + height / 2])
#
#
# def A_inv(val):
#     global width, height
#     return np.array([val[0] - width / 2, -(val[1] - height / 2)])
#
#
# def A_many(vals):
#     return [A(v) for v in vals]   # Galaxy brain function
#
#
# # Keyhandling (modify + redraw call)
# def handle_keys(keys_pressed):
#     pass
#
#
# # Mesh data
# positions = []
# velocities = []
# forces = []
# neighbors = []
#
# # Mesh information (input)
# filename = 'cube.obj'  #'lowpolyface.obj'
# pinned = [1, 5, 3]  #[168, 166, 180, 171, 172, 170, 178, 164, 169, 185, 176, 182, 165, 179, 174, 175, 173, 181, 167, 183, 177, 184]
# stiff = 100
# h = 0.00005
# drag = 0.00009
#
# # Load OBJ file + fill adjacency list
# # 1. Load the raw vertices, edges, and faces list.
# faces = []
# edges = set()
# delimiter = '/'
# with open(filename) as f:
#     for line in f:
#         kind = line[0]
#         if kind == 'v':
#             pass
#             positions.append([float(l) for l in line.rsplit()[1:]])
#         elif kind == 'f':
#             indices = None
#             if delimiter in line:
#                 indices = [int(l[:l.find(delimiter)]) - 1 for l in line.rsplit()[1:]]
#             else:
#                 indices = [int(l) - 1 for l in line.rsplit()[1:]]
#
#             faces.append(indices)
#
#             # Add edges for each face
#             num_vertices = len(indices)
#             for i in range(num_vertices):
#                 # Create an edge (vi, vj), ensuring vi < vj for uniqueness
#                 vi = indices[i]
#                 vj = indices[(i + 1) % num_vertices]  # Wrap around to the first vertex
#                 edge = (min(vi, vj), max(vi, vj))
#                 edges.add(edge)
#
# edges = list(edges)
#
# # 2. Build adjacency list
# assert len(faces[0]) == 3  # should be a trimesh
# neighbors = [[] for _ in positions]
# for edge in edges:
#     v1, v2 = edge
#     if v2 not in neighbors[v1]:
#         neighbors[v1].append(v2)
#     if v1 not in neighbors[v2]:
#         neighbors[v2].append(v1)
#
# # 3. Randomize positions of non-pinned verts on screen, and their velocities
# np.random.seed(42)
# velocities = [0 for _ in positions]
# forces = [0 for _ in positions]
# for i in range(len(positions)):
#     if i not in pinned:
#         x_max, y_max = width * 0.5, height * 0.5
#         positions[i] = np.random.rand(2) * [x_max, y_max] - [x_max / 2, y_max / 2]
#         maxvel = 2
#     else:
#         angle = np.pi * 2.0 * (pinned.index(i) / float(len(pinned)))
#         positions[i] = np.array([np.cos(angle), np.sin(angle)]) * min(width, height) / 2 * 0.9
#
#
# # Additional vars
# colors = {
#     'white': np.array([255., 255., 255.]),
#     'black': np.array([0., 0., 0.]),
#     'red': np.array([255, 66, 48]),
#     'blue': np.array([30, 5, 252]),
#     'fullred': np.array([255, 0, 0]),
#     'fullblue': np.array([0, 0, 255]),
#     'START': np.array([255, 255, 255])
# }
#
#
# def main():
#     # Pre-gameloop stuff
#     run = True
#     clock = pygame.time.Clock()
#
#     # Game loop
#     count = 0
#     while run:
#         # Reset / increment stuff
#         count += 1
#         window.fill((0, 0, 0))
#         # Force accumulation
#         for i in range(len(positions)):
#             if i not in pinned:
#             # if i == 0:
#                 for neigh in neighbors[i]:
#                     forces[i] += stiff * (positions[neigh] - positions[i])
#                     forces[i] += -drag * np.sign(velocities[i]) * np.power(velocities[i], 2.0)
#         # Velocity update
#         for i in range(len(positions)):
#             velocities[i] += h * forces[i]
#         # Position update
#         for i in range(len(positions)):
#             positions[i] += h * velocities[i]
#
#
#         # Draw edges
#         for e in edges:
#             v1, v2 = positions[e[0]], positions[e[1]]
#             pygame.draw.line(window, colors['blue'], A(v1), A(v2), width=1)
#
#         # Draw nodes
#         for i, p in enumerate(positions):
#             pygame.draw.circle(window, colors['white'] if i not in pinned else colors['red'], A(p), 3, width=0)
#
#
#
#         # End run (1. Tick clock, 2. Save the frame, and 3. Detect if window closed)
#         pygame.display.update()
#         clock.tick(60)  # FPS
#         count += 1
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 run = False
#                 pygame.quit()
#
#
# if __name__ == '__main__':
#     main()