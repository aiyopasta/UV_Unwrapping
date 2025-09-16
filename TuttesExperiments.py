import numpy as np
import pygame
import os
import trimesh
import scipy as sp
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)

# Save the animation? TODO: Make sure you're saving to correct destination!!
save_anim = False

# Pygame + gameloop setup
width = 1080
height = 1080
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


# Mesh information (input)

# FACE
mesh_file = 'meshes/SUBDIVIDED_FACE_10K.obj'
pinned = [128, 4687, 791, 2739, 418, 4287, 2339, 6235, 142, 6234, 2338, 4286, 417, 4487, 2539, 6435, 143, 6433, 2537, 4485, 412, 3912, 1964, 5860, 149, 5858, 1962, 3910, 409, 4282, 2334, 6230, 134, 6237, 2341, 4289, 425, 3917, 1969, 5865, 133, 6014, 2118, 4066, 422, 2937, 989, 4885, 137, 5021, 1125, 3073, 407, 3437, 1489, 5385, 147, 5463, 1567, 3515, 414, 3319, 1371, 5267, 146, 5265, 1369, 3317, 410, 2933, 985, 4881, 145, 4883, 987, 2935, 415, 2737, 789, 4685, 144, 4683, 787, 2735, 408, 3909, 1961, 5857, 141, 5862, 1966, 3914, 419, 3716, 1768, 5664, 140, 5663, 1767, 3715, 416, 2738, 790, 4686, 136, 4690, 794, 2742, 423, 4402, 2454, 6350, 135, 6351, 2455, 4403, 424, 3717, 1769, 5665, 132, 5667, 1771, 3719, 426, 3567, 1619, 5515, 131, 5465, 1569, 3517, 421, 3322, 1374, 5270, 138, 5169, 1273, 3221, 420, 3516, 1568, 5464, 139, 5462, 1566, 3514, 413, 3713, 1765, 5661, 148, 5660, 1764, 3712, 411, 3318, 1370, 5266, 130, 5156, 1260, 3208, 598, 3422, 1474, 5370, 427, 2938, 990, 4886, 129, 5038, 1142, 3090, 428, 2743, 795, 4691]

# BUNNY
# mesh_file = 'meshes/BUNNY_OPENED.obj'
# pinned = [915, 2503, 1057, 1188, 2504, 1207, 2508, 2506, 1326, 1371, 2507, 1285, 1351, 2505]

# ...
# mesh_file = 'meshes/CUT_HEAD_AT_CHEEK.obj'
# pinned = [1377, 1378, 1567, 1555, 1857, 1549, 1551, 1552, 1553, 1640, 1639, 1628, 1546]

# DOESN'T WORK
# mesh_file = 'meshes/CUT_SKULL.obj'
# pinned = [4645, 4666, 4696, 4699, 4725, 4787, 4871, 4910, 4957, 4863, 4781, 4721, 4667, 4668]

# mesh_file = 'meshes/CUT_CAESAR.obj'
# pinned = [69, 70, 2271, 2543, 106, 2755, 2688, 1970, 2799, 2405, 2797, 2789, 2405, 2799, 1970, 2688, 2755, 106, 2543, 2271, 70]

# mesh_file = 'meshes/CUT_CAESAR_MASK.obj'
# pinned = [93, 1225, 3569, 3555, 1549, 3591, 3587, 3565, 3519, 3604, 3597, 3538, 3523, 2671, 3548, 3554, 3492, 3520, 3522, 3161, 3287, 2618, 3528, 3580, 3550, 3560, 3571, 3545, 3563, 2106, 1484, 1683, 2353, 3531, 3584, 3521, 3559, 3258, 2773, 2661, 3572, 3245, 3223, 3222, 3003, 3600, 3526, 3534, 2074, 1372, 3598, 958, 3612, 3558, 3541, 3414, 3539, 2227, 3544, 3606, 3182, 3529, 3542, 2063, 3517, 3515, 3599, 3585, 3575, 3029, 3028, 3579, 3570, 3536, 3540, 786, 3594, 3602, 3586, 3553, 2541, 3556, 3566, 3183, 3564, 3583, 3561, 2982, 3530, 3537, 3557, 3516, 3608, 3518, 2640, 3535, 3595, 3593, 2788, 2108, 3402, 3551, 3272, 2237, 3613, 3562, 3592, 3549, 3525, 3543, 3605, 3607, 3601, 3533, 3590, 3209, 3527, 3588, 3610, 3611, 1506, 3581, 3567, 3574, 3568, 3582, 3603, 2101, 3576, 3532, 3097, 1641, 3547, 2131, 3573, 3596, 3589, 1741, 3524, 2337, 3609, 3215, 3546, 2954, 3577, 3578, 3552, 2504]

mesh_file = 'meshes/CUT_CAESAR_MASK_CUT_HOLES.obj'  # Pins below: (Main, eye, mouth, nose)
# pinned = [92, 1188, 3509, 3495, 1507, 3531, 3527, 3505, 3459, 3544, 3537, 3478, 3463, 2614, 3488, 3494, 3432, 3460, 3462, 3102, 3228, 2561, 3468, 3520, 3490, 3500, 3511, 3485, 3503, 2059, 1443, 1638, 2302, 3471, 3524, 3461, 3499, 3199, 2714, 2604, 3512, 3186, 3164, 3163, 2944, 3540, 3466, 3474, 2027, 1334, 3538, 925, 3552, 3498, 3481, 3355, 3479, 2177, 3484, 3546, 3123, 3469, 3482, 2016, 3457, 3455, 3539, 3525, 3515, 2970, 2969, 3519, 3510, 3476, 3480, 759, 3534, 3542, 3526, 3493, 2489, 3496, 3506, 3124, 3504, 3523, 3501, 2923, 3470, 3477, 3497, 3456, 3548, 3458, 2583, 3475, 3535, 3533, 2729, 2061, 3343, 3491, 3213, 2187, 3553, 3502, 3532, 3489, 3465, 3483, 3545, 3547, 3541, 3473, 3530, 3150, 3467, 3528, 3550, 3551, 1465, 3521, 3507, 3514, 3508, 3522, 3543, 2054, 3516, 3472, 3038, 1599, 3487, 2082, 3513, 3536, 3529, 1695, 3464, 2286, 3549, 3156, 3486, 2895, 3517, 3518, 3492, 2452]
pinned = [210, 2447, 985, 1575, 1418, 1319, 872, 843, 262, 333, 334, 834, 467, 468, 259, 260, 578, 579, 886, 887, 711, 296, 298, 775]
# pinned = [54, 55, 3594, 1175, 3599, 3584, 589, 148, 3592, 3575, 3574, 3600, 662, 1111, 3585, 569, 878, 483, 695, 3578, 3579, 588, 131, 3593, 1177, 3583, 3582, 3588, 3595, 3577, 3576, 149, 696, 3580, 153, 3573, 3572, 281, 280, 3591, 3598, 2395, 1752, 1753, 3586, 3587, 388, 1360, 3590, 3597, 190, 673, 3581, 152, 3596, 3589, 724, 443, 1020, 771]
# pinned = [3554, 3559, 3555, 3560, 3558, 3557, 3562, 3556, 3561, 3563]
# pinned = [1399, 1903, 1902]
stiff = 2000
h = 0.01

mesh = trimesh.load(mesh_file)
faces = mesh.faces
edges = mesh.edges_unique
positions = np.zeros((len(mesh.vertices), 2))

# 2. Override positions with randomized values if unpinned, else circular values.
np.random.seed(42)
angle_offset = np.radians(-120.) - np.radians(30.)
for i in range(positions.shape[0]):
    if i not in pinned:
        x_max, y_max = width * 0.9, height * 0.9
        positions[i] = np.random.rand(2) * [x_max, y_max] - [x_max / 2, y_max / 2]
        maxvel = 2
    else:
        positions[i] = rect(pinned.index(i) / float(len(pinned)), width * 0.9, width * 0.9)
        # angle = (np.pi * 2.0 * (pinned.index(i) / float(len(pinned)))) + angle_offset
        # positions[i] = np.array([np.cos(angle), np.sin(angle)]) * min(width, height) / 2 * 0.9

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
    global positions, calm, pinned
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
        anchor = positions[list(pinned), :].mean(axis=0)
        positions = ((positions - anchor) @ rot_mat(theta_delta).T) + anchor
    if keys_pressed[pygame.K_k]:
        anchor = positions[list(pinned), :].mean(axis=0)
        positions = ((positions - anchor) @ rot_mat(-theta_delta).T) + anchor



    # For saving UV map
    elif keys_pressed[pygame.K_s]:
        with open("face_uv.obj", 'w') as file:
            for p in positions:
                # Write each vertex with x, y from the list and z as 0
                file.write(f'v {p[0]} {p[1]} 0\n')


# Precompute drawing things that don't change (TODO: Ideally, I'd use something like Pyglet for fast draw)
# (a) Edge coloring
edge_cols = [lerp(i / len(edges), colors['lightblue'], colors['lightgreen']) for i in range(len(edges))]

# (b) Quicker dot drawing ("draw()" is slow, so we draw it to a single surface, and blit that surface elsewhere,
#     which is faster apparently, because blitting is implemented in C. It's kinda like instanced rendering ig.)
def make_dot(color, radius=1):
    surf = pygame.Surface((2 * radius + 1, 2 * radius + 1), pygame.SRCALPHA)  # a tinyass surface for this dot
    pygame.draw.circle(surf, color, (radius, radius), radius)
    return surf

# Actually compute and stuff the two surfaces (white and red) for each of the two possible dots
dot_white = make_dot(colors['white'])
dot_red   = make_dot(colors['red'])

# Instead of O(n) lookup test, reduce to O(1)
pinned = set(pinned)

# Evolve step (main springy code)
def simulate():
    global positions, velocities, pinned
    # Compute new velocities
    # 1. Reshape old velocity + position vectors
    v0 = velocities.reshape(-1, 1)
    p = positions.reshape(-1, 1)
    # 2. Compute new velocities
    v = v0 + (M @ (p + (h * v0)))
    v = v.reshape(len(positions), 2)
    # 3. Update velocities of only un-pinned vertices
    for row, vel in enumerate(v):
        if row not in pinned:
            velocities[row] = vel
    # 4. Update positions based on velocities
    positions += h * velocities

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

        # Simulate step
        simulate()

        # Fast(er) Drawing ————
        # Draw edges
        P = A_many(positions)  # makes it much faster!!
        line = pygame.draw.line  # wow predefining this is actually slightly faster.
        for i, (i1, i2) in enumerate(edges):
            line(window, edge_cols[i], P[i1], P[i2], width=1)

        # Draw dots
        blit = window.blit  # wow predefining this is actually slightly faster.
        for i, pt in enumerate(P):
            spr = dot_white if i not in pinned else dot_red
            blit(spr, spr.get_rect(center=(int(pt[0]), int(pt[1]))))

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
        else:
            print(clock.get_fps())

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