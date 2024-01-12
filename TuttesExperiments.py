# Interesting find: As I increase the time-step, the drawing gets more and more concentrated towards the center.

import numpy as np
import pygame
import os
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)

# Save the animation? TODO: Make sure you're saving to correct destination!!
save_anim = False

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


# Helper Functions
def lerp(u, initial, final):
    return ((1. - u) * initial) + (u * final)


# Mesh Information (input)
filename = 'cut_dolphin.obj'
pinned = [52, 82, 83, 81]

# This is for "THE ACTUAL FACE TO USE".obj, it's the true boundary.
#[128, 418, 142, 417, 143, 412, 149, 409, 134, 425, 133, 422, 137, 407, 147, 414, 146, 410, 145, 415, 144, 408, 141, 419, 140, 416, 136, 423, 135, 424, 132, 426, 131, 421, 138, 420, 139, 413, 148, 411, 130, 598, 427, 129, 428]


# # Mesh information (input)
# filename = 'lowpolyface7.obj'  # use face 7
# pinned = [67, 482, 207]  #[128, 418, 142, 417, 143, 412, 149, 409, 134, 425, 133, 422, 137, 407, 147, 414, 146, 410, 145, 415, 144, 408, 141, 419, 140, 416, 136, 423, 135, 424, 132, 426, 131, 421, 138, 420, 139, 413, 148, 411, 130, 598, 427, 129, 428]
#
# # FOR FACE 7
# # Random loop [21, 151, 48, 237, 64, 270, 101, 262, 104, 258, 478, 177, 464, 228, 472, 252, 608, 657, 638, 648, 636, 646, 632, 643, 631, 675, 615, 41, 163, 42, 205]
# # FROM MOUTH [629, 641, 630, 645, 633, 644, 654, 655, 653, 637, 647, 639, 649, 640, 651, 634, 650, 635, 652, 638, 648, 636, 646, 632, 643, 631, 642]
# # FROM EYE [112, 367, 125, 376, 119, 364, 123, 365, 114, 369, 122, 363, 117, 375, 126, 370]
# # DUMB TRIANGLE [67, 482, 207]
# # FROM TRUE BOUNDARY [128, 418, 142, 417, 143, 412, 149, 409, 134, 425, 133, 422, 137, 407, 147, 414, 146, 410, 145, 415, 144, 408, 141, 419, 140, 416, 136, 423, 135, 424, 132, 426, 131, 421, 138, 420, 139, 413, 148, 411, 130, 598, 427, 129, 428]
#
# # FOR FACE 6, 5, 4 ...
# #[128, 418, 142, 417, 143, 412, 149, 409, 134, 425, 133, 422, 137, 407, 147, 414, 146, 410, 145, 415, 144, 408, 141, 419, 140, 416, 136, 423, 135, 424, 132, 426, 131, 421, 138, 420, 139, 413, 148, 411, 130, 598, 427, 129, 428]
# #[143, 465, 157, 464, 158, 459, 164, 456, 149, 472, 148, 469, 152, 454, 162, 461, 161, 457, 160, 462, 159, 455, 156, 466, 155, 463, 151, 470, 150, 471, 147, 473, 146, 468, 153, 467, 154, 460, 163, 458, 145, 474, 144, 475]
# #[163, 525, 177, 524, 178, 519, 184, 516, 169, 532, 168, 529, 172, 514, 182, 521, 181, 517, 180, 522, 179, 515, 176, 526, 175, 523, 171, 530, 170, 531, 167, 533, 166, 528, 173, 527, 174, 520, 183, 518, 165, 534, 164, 535]
# #[163, 525, 177, 524, 178, 519, 184, 516, 169, 532, 168, 529, 172, 514, 182, 521, 181, 517, 180, 522, 179, 515, 176, 526, 175, 523, 171, 530, 170, 531, 167, 533, 166, 528, 173, 527, 174, 520, 183, 518, 165, 534, 164, 535]
# #[163, 177, 178, 184, 169, 168, 172, 182, 181, 180, 179, 176, 175, 171, 170, 167, 166, 173, 174, 183, 165, 164]
#  #[168, 166, 180, 171, 172, 170, 178, 164, 169, 185, 176, 182, 165, 179, 174, 175, 173, 181, 167, 183, 177, 184]



stiff = 200
h = 0.005 #0.02
# drag = 0.00009

# Load OBJ file + fill adjacency list
# 1. Load the raw vertices, edges, and faces list.
positions = []
faces = []
edges = set()
delimiter = '//'
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
angle_offset = np.radians(-120.) - np.radians(30.)
for i in range(len(positions)):
    if i not in pinned:
        x_max, y_max = width * 0.9, height * 0.9
        positions[i] = np.random.rand(2) * [x_max, y_max] - [x_max / 2, y_max / 2]
        maxvel = 2
    else:
        angle = (np.pi * 2.0 * (pinned.index(i) / float(len(pinned)))) + angle_offset
        positions[i] = np.array([np.cos(angle), np.sin(angle)]) * min(width, height) / 2 * 0.9

# # 2. TODO (for now) Override positions copied over from other file
# positions = np.loadtxt('scratch.txt')

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


# Keyhandling (for zooming in and out)
def handle_keys(keys_pressed):
    global positions

    # For zooming in / out
    if keys_pressed[pygame.K_p]:
        positions *= 1.5
    elif keys_pressed[pygame.K_o]:
        positions /= 1.5

    # For saving UV map
    elif keys_pressed[pygame.K_s]:
        with open("face_uv.obj", 'w') as file:
            for p in positions:
                # Write each vertex with x, y from the list and z as 0
                file.write(f'v {p[0]} {p[1]} 0\n')


def main():
    global positions, velocities

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

    # Game loop
    count = 0
    while run:
        # Reset stuff
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
        for i, e in enumerate(edges):
            v1, v2 = positions[e[0]], positions[e[1]]
            u = i / len(edges)
            pygame.draw.line(window, lerp(u, colors['blue'], colors['red']), A(v1), A(v2), width=1)

        # Draw nodes
        for i, p in enumerate(positions):
            pygame.draw.circle(window, colors['white'] if i not in pinned else colors['red'], A(p), 2, width=0)


        # Handle keys
        keys_pressed = pygame.key.get_pressed()
        handle_keys(keys_pressed)


        # End run (1. Tick clock, 2. Save the frame, and 3. Detect if window closed)
        pygame.display.update()
        clock.tick(60)  # FPS
        count += 1
        if save_anim:
            pygame.image.save(window, path_to_save + '/frame' + str(count) + '.png')
            print('Saved frame ' + str(count))

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