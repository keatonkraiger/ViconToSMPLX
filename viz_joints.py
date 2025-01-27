import numpy as np
import torch
from smplx import SMPLXLayer
from utils.vicon_mapping import ESTIMATED_TO_SMPLX
from utils.rotation_conversion import aa2rot_torch

mapping= {
        "C7": 3832,
        "CLAV": 5533,
        "LANK": 5882,
        "LBHD": 2026,
        "LBWT": 5697,
        "LELB": 4302,
        "LFHD": 707,
        "LFIN": 4788,
        "LFRM": 4198,
        "LFWT": 3486,
        "LHEE": 8846,
        "LIWR": 4726,
        "LKNE": 3682,
        "LOWR": 4722,
        "LSHO": 4481,
        "LTHI": 4088,
        "LTIB": 3745,
        "LTOE": 5787,
        "LUPA": 4030,
        "RANK": 8576,
        "RBAK": 6127,
        "RBHD": 3066,
        "RBWT": 8391,
        "RELB": 7040,
        "RFHD": 2198,
        "RFIN": 7524,
        "RFRM": 6942,
        "RFWT": 6248,
        "RHEE": 8634,
        "RIWR": 7462,
        "RKNE": 6443,
        "ROWR": 7458,
        "RSHO": 6627,
        "RTHI": 6832,
        "RTIB": 6503,
        "RTOE": 8481,
        "RUPA": 6777,
        "STRN": 5531,
        "T10": 5623
      }

def save_highlighted_mapping_with_labels(model_path, marker_mapping, gender='neutral', output_obj="highlighted_mapping.obj"):
    from smplx import SMPLXLayer
    import numpy as np

    # Load SMPL-X model
    smplx_model = SMPLXLayer(model_path=model_path, gender=gender)

    # Generate neutral pose parameters
    params = {
        "global_orient": torch.zeros(1, 3),
        "body_pose": torch.zeros(1, 21, 3),
        "left_hand_pose": torch.zeros(1, 15, 3),
        "right_hand_pose": torch.zeros(1, 15, 3),
        "jaw_pose": torch.zeros(1, 3),
        "leye_pose": torch.zeros(1, 3),
        "reye_pose": torch.zeros(1, 3),
        "expression": torch.zeros(1, 10),
        "betas": torch.zeros(1, 10),
        "transl": torch.zeros(1, 3),
    }

    # Convert axis-angle to rotation matrices
    global_orient_rot = aa2rot_torch(params["global_orient"]).reshape(1, 1, 3, 3)
    body_pose_rot = aa2rot_torch(params["body_pose"].reshape(-1, 3)).reshape(1, 21, 3, 3)

    # Forward pass through SMPL-X to get body vertices
    smplx_output = smplx_model(
        global_orient=global_orient_rot,
        body_pose=body_pose_rot,
        transl=params["transl"],
        betas=params["betas"],
    )
    vertices = smplx_output.vertices.detach().cpu().numpy()[0]  # [num_vertices, 3]
    faces = smplx_model.faces

    # Add large dots for each marker
    dot_radius = 0.02  # Radius of the dots
    dot_segments = 8   # Sphere resolution for the dots
    dot_vertices = []
    dot_faces = []
    color_offset = len(vertices)  # Keep track of indices for faces

    def create_sphere(center, radius, segments):
        """Generate vertices and faces for a sphere."""
        sphere_verts = []
        sphere_faces = []
        for i in range(segments + 1):
            theta = np.pi * i / segments
            for j in range(segments):
                phi = 2 * np.pi * j / segments
                x = center[0] + radius * np.sin(theta) * np.cos(phi)
                y = center[1] + radius * np.sin(theta) * np.sin(phi)
                z = center[2] + radius * np.cos(theta)
                sphere_verts.append([x, y, z])
                if i > 0 and j > 0:
                    p1 = i * segments + j
                    p2 = (i - 1) * segments + j
                    p3 = p1 + 1
                    p4 = p2 + 1
                    sphere_faces.append([p1, p2, p3])
                    sphere_faces.append([p3, p2, p4])
        return sphere_verts, sphere_faces

    for marker, vertex_idx in marker_mapping.items():
        marker_position = vertices[vertex_idx]
        sphere_verts, sphere_faces = create_sphere(marker_position, dot_radius, dot_segments)

        # Append sphere vertices and offset faces
        dot_vertices.extend(sphere_verts)
        dot_faces.extend([[f[0] + color_offset, f[1] + color_offset, f[2] + color_offset] for f in sphere_faces])
        color_offset += len(sphere_verts)

    # Save OBJ with dots and original mesh
    with open(output_obj, "w") as f:
        # Write original vertices
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]} 0.8 0.8 0.8\n")  # Grey body color

        # Write dot vertices
        for v in dot_vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]} 1.0 0.0 0.0\n")  # Red for dots

        # Write original faces
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

        # Write dot faces
        for face in dot_faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print(f"Visualization with large colored dots saved to {output_obj}")



save_highlighted_mapping_with_labels('body_models/smplx',mapping, gender='neutral', output_obj="highlighted_mapping.obj")