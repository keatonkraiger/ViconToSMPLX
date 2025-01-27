import os.path as osp
import torch
import numpy as np
import numpy as np

def save_as_obj(file_path, vertices, faces):
    """
    Save SMPL-X model as an OBJ file.
    Args:
        file_path (str): Path to save the OBJ file.
        vertices (numpy.ndarray): Array of vertices with shape [V, 3].
        faces (numpy.ndarray): Array of faces with shape [F, 3].
    """
    # Scale vertices to meters
    vertices /= 1000.0

    # Recenter vertices
    centroid = vertices.mean(axis=0)
    vertices -= centroid

    # Write to OBJ
    with open(file_path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces + 1:  # OBJ format is 1-indexed
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")


def move_to_cpu(data):
    """
    Recursively move any torch.Tensor inside a dict (or nested structure)
    to CPU. If you only have a flat dict with Tensors, this simple version
    is enough.
    """
    out = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu()
        else:
            out[k] = v
    return out

    
def write_smplx(smplx_params,out_path):
    """
    smplx_params: smplx params of all frames. Move to cpu if not already
    """
    #nf = smplx_params['body_pose'].shape[0]
    #smplx = move_to_cpu(smplx_params) 
    #np.savez(osp.join(out_path), **smplx)
     
    smplx = {
        'body_pose': smplx_params['body_pose'],             # [F, J_body, 3]
        'lhand_pose': smplx_params['lhand_pose'],           # [F, J_lhand, 3]
        'rhand_pose': smplx_params['rhand_pose'],           # [F, J_rhand, 3]
        'jaw_pose': smplx_params['jaw_pose'],               # [F, 3]
        'leye_pose': smplx_params['leye_pose'],             # [F, 3]
        'reye_pose': smplx_params['reye_pose'],             # [F, 3]
        'betas': smplx_params['betas'],                     # [F, 10] or [1, F, 10]
        'expression': smplx_params['expression'],           # [F, 10]
        'global_orient': smplx_params['global_orient'],     # [F, 3]
        'transl': smplx_params['transl']                    # [F, 3]
    }
    smplx_params_cpu = move_to_cpu(smplx)
    np.savez(out_path, **smplx_params_cpu)
def read_smplx(smplx_path):
    """
        smplx_path: path to smplx.npz file
        return:
                params: dict of np.array

    """
    params=np.load(smplx_path,allow_pickle=True)
    # restore to dict of np.array
    params={
        key:np.array(params[key].tolist()) for key in params.files
    }
    return params
