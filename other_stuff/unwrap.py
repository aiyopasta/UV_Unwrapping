from tkinter import *
import numpy as np

# Window size
window_w = 1720
window_h = 1080
# np.set_printoptions(suppress=True)

# Tkinter Setup
root = Tk()
root.title("Simulator")
root.attributes("-topmost", True)
root.geometry(str(window_w) + "x" + str(window_h))  # window size hardcoded
w = Canvas(root, width=window_w, height=window_h)
w.pack()

# Key handling function
def vanilla_key_pressed(event):
    global rho, theta, phi, focus, w

    m = 30
    drho, dphi, dtheta, dfocus = 10, np.pi/m, np.pi/m, 10
    if event.char == 'a':
        theta -= dtheta

    if event.char == 'd':
        theta += dtheta

    if event.char == 'w':
        phi -= dphi

    if event.char == 's':
        phi += dphi

    if event.char == 'p':
        rho -= drho

    if event.char == 'o':
        rho += drho

    if event.char == 'k':
        focus -= dfocus

    if event.char == 'l':
        focus += dfocus

    if event.char == 'm':
        w.bind("<KeyPress>", speedy_key_pressed)


def speedy_key_pressed(event):
    global v_rho, v_theta, v_phi, focus, w

    max_clicks = 10
    m = 800
    d2rho, d2phi, d2theta, dfocus = 3, np.pi / m, np.pi / m, 10
    if event.char == 'a':
        v_theta = max(v_theta - d2theta, -d2theta*max_clicks)

    if event.char == 'd':
        v_theta = min(v_theta + d2theta, d2theta*max_clicks)

    if event.char == 'w':
        v_phi = max(v_phi - d2phi, -d2phi*max_clicks)

    if event.char == 's':
        v_phi = min(v_phi + d2phi, d2phi*max_clicks)

    if event.char == 'p':
        v_rho = max(v_rho - d2rho, -d2rho*max_clicks/2)

    if event.char == 'o':
        v_rho = min(v_rho + d2rho, d2rho*max_clicks/2)

    if event.char == 'k':
        focus -= dfocus

    if event.char == 'l':
        focus += dfocus

    # Change mode
    if event.char == 'm':
        v_rho, v_theta, v_phi = 0, 0, 0
        w.bind("<KeyPress>", vanilla_key_pressed)


# Key binding
w.bind("<KeyPress>", speedy_key_pressed)
w.bind("<1>", lambda event: w.focus_set())
w.pack()

# ye parameters (rho = distance from origin, phi = angle from world's +z-axis, theta = angle from world's +x-axis)
rho, theta, phi = 700., np.pi/4, np.pi/4  # These provide location of the eye.
v_rho, v_theta, v_phi = 0, 0, 0
focus = 500.  # Distance from eye to near clipping plane, i.e. the screen.
far_clip = rho * 3  # Distance from eye to far clipping plane
assert far_clip > focus


# Input: A point in 3D world space. Output: Corresponding point in range [-width/2, width/2]x[-height/2, height/2].
def world_to_plane(v):
    global rho, theta, phi, focus
    # -1. Adjust focus based on
    # 0. Turn vector into a homogeneous one.
    v = np.append(v, 1)
    # 1. Convert vector from world into camera space
    # a) Get camera basis vectors in terms of world coordinates (x right, y up, z out of page).
    xhat = np.array([-np.sin(theta), np.cos(theta), 0, 0])
    yhat = -np.array([np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), -np.sin(phi), 0])
    zhat = np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi), 0])
    # b) Build the 4th column of the matrix
    cam_pos = np.append((rho * zhat)[:3], 1)
    # c) Construct the matrix and do the conversion
    world_to_cam = np.linalg.inv(np.array([xhat, yhat, zhat, cam_pos]).T)
    v_cam = np.dot(world_to_cam, v)
    # 2. Convert from camera space to screen space (using similar triangles math)
    cam_to_screen = np.array([[-focus, 0, 0, 0],
                              [0, -focus, 0, 0],
                              [0, 0, -far_clip/(-far_clip+focus), (-far_clip*focus)/(-far_clip+focus)],
                              [0, 0, 1, 0]])
    v_screen = np.dot(cam_to_screen, v_cam)
    v_screen /= v_screen[3]  # division by z
    return (v_screen[:2] * np.array([1, -1])) + np.array([window_w/2, window_h/2])


# 1000 IQ Genius Function
def list_world_to_plane(l):
    new_l = []
    for v in l:
        new_l.extend(world_to_plane(v).tolist())
    return new_l


# Camera position
def campos():
    global rho, phi, theta
    return rho * np.array([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])


# Mesh storage
# We'll save: 1) A list of actual vertex positions,
#             2) A list of tuples representing edges. NOTE: THESE ARE DOUBLE COUNTED!
#             3) A list of 3 tuples representing faces (oriented counter-clockwise).
#             4) An incidence vertex / edge matrix, and...
#             5) ... an incidence edge / face matrix. See https://youtu.be/HePDHsp8spU?t=1924.
vertices = []
edges = []  # these are directed edges, and they're double counted!
faces = []
vert_edge_mat = None
edge_face_mat = None


# Show / hide display objects
show_axes = True
show_obj = True


# Main function
def run():
    global rho, phi, theta, v_rho, v_theta, v_phi, show_axes, vertices, edges, faces, vert_edge_mat, edge_face_mat
    w.configure(background='black')

    # Load the obj file
    # 1. Load the raw vertices, edges, and faces list.
    with open('heart.obj') as f:
        for line in f:
            type = line[0]
            if type == 'v':
                vertices.append([float(l) for l in line.rsplit()[1:]])
            elif type == 'f':
                indices = None
                if '//' in line:
                    indices = [int(l[:l.find('//')])-1 for l in line.rsplit()[1:]]
                else:
                    indices = [int(l)-1 for l in line.rsplit()[1:]]

                edges.append((indices[0], indices[1]))
                edges.append((indices[1], indices[2]))
                edges.append((indices[2], indices[0]))
                faces.append(indices)

    # 2. Create the two incidence matrices. (rows=edges, columns=vertex neighbors)
    for e in edges:
        row = [int(idx in e) for idx in range(len(vertices))]
        if vert_edge_mat is None:
            vert_edge_mat = np.array(row)
        else:
            vert_edge_mat = np.vstack([vert_edge_mat, row])

    for f in faces:
        row = [int(set(e).issubset(set(f))) for e in edges]
        if edge_face_mat is None:
            edge_face_mat = np.array(row)
        else:
            edge_face_mat = np.vstack([edge_face_mat, row])


    # Main loop
    while True:
        w.delete('all')
        # Camera Velocity Update
        rho += v_rho
        phi += v_phi
        theta += v_theta

        # 3D Axes Drawing ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
        colors = ['red', 'blue', 'purple']
        mag = 200
        direction_vectors = [np.array([mag, 0, 0]), np.array([0, mag, 0]), np.array([0, 0, mag])]
        if show_axes:
            for i, v_i in enumerate(direction_vectors):
                tile_center = world_to_plane(v_i)
                w.create_line(window_w/2, window_h/2, tile_center[0], tile_center[1], fill=colors[i])

        # Draw the OBJ on screen (face by face)
        for f in faces:
            points = []
            rgb = None
            for i in range(3):
                pt3d = np.array(vertices[f[i]])
                points.extend(world_to_plane(pt3d))
                percent = np.power(1.0 - min(np.linalg.norm(campos() - pt3d) / (2 * rho), 1.0), 1.5)
                rgb = tuple([int(i) for i in percent * np.array([255, 0, 0])])

            w.create_polygon(*points, fill='', outline=_from_rgb(rgb))


        # End run
        w.update()


# From https://stackoverflow.com/questions/51591456/can-i-use-rgb-in-tkinter
def _from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb


# Main function
if __name__ == '__main__':
    run()

# Necessary line for Tkinter
mainloop()
