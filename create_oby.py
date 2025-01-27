import os
import numpy as np
import torch

# SMPL-X model
import smplx
# ------------------------------------------------------------------------------
# EDIT THESE PATHS:
MODEL_PATH = "body_models"   # Folder with SMPLX_NEUTRAL.npz, etc.
NPZ_PATH = "output/Subject1/1.npz"  # The file you saved with write_smplx
OUTPUT_FOLDER = "output_objs"
GENDER = "neutral"  # or "male", "female" if you have those models
NUM_BETAS = 10      # Typically 10 shape coeffs, adjust if you used more
NUM_EXPR = 10       # Typically 10 expression coeffs, or 50 if you used more
# ------------------------------------------------------------------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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

def main():
    # 1. Load the SMPL-X model
    smplx_model = smplx.create(
        model_path=MODEL_PATH,
        model_type='smplx',
        gender=GENDER,
        num_betas=NUM_BETAS,
        num_expression_coeffs=NUM_EXPR,
        use_pca=False  # assume you have explicit hand poses, not PCA-based
    )
    
    # 2. Load the .npz that contains SMPL-X params over multiple frames
    data = read_smplx(NPZ_PATH)

    # Pull arrays from the dictionary. 
    # (Make sure these keys match exactly your saved dictionary!)
    body_pose_np = data['body_pose']       # shape (F, 63)
    lhand_pose_np = data['lhand_pose']     # shape (F, 45)
    rhand_pose_np = data['rhand_pose']     # shape (F, 45)
    jaw_pose_np = data['jaw_pose']         # shape (F, 3)
    leye_pose_np = data['leye_pose']       # shape (F, 3)
    reye_pose_np = data['reye_pose']       # shape (F, 3)
    betas_np = data['betas']           # shape this loads a [1,F,10] array.
    expr_np = data['expression']           # shape (F, NUM_EXPR)
    global_orient_np = data['global_orient']  # shape (F, 3)
    transl_np = data['transl']             # shape (F, 3)

    # Number of frames
    num_frames = body_pose_np.shape[0]

    # Convert betas to Torch once (betas typically do NOT change per-frame)
    # Set betas to tensor of zeros
    betas_torch = torch.tensor(betas_np[0], dtype=torch.float32).unsqueeze(0)
    # 3. Iterate over each frame, run SMPL-X forward, and export an OBJ
    faces = smplx_model.faces  # (F, 3) for the SMPL-X mesh topology
    for frame_idx in range(num_frames):
        # Convert each parameter to a 1 x D Torch tensor
        transl = transl_np[frame_idx]
        mean_translation = np.mean(transl, axis=0)
        transl-= mean_translation 
        
        body_pose_torch = torch.tensor(body_pose_np[frame_idx], dtype=torch.float32).unsqueeze(0)
        lhand_pose_torch = torch.tensor(lhand_pose_np[frame_idx], dtype=torch.float32).unsqueeze(0)
        rhand_pose_torch = torch.tensor(rhand_pose_np[frame_idx], dtype=torch.float32).unsqueeze(0)
        jaw_pose_torch = torch.tensor(jaw_pose_np[frame_idx], dtype=torch.float32).unsqueeze(0)
        leye_pose_torch = torch.tensor(leye_pose_np[frame_idx], dtype=torch.float32).unsqueeze(0)
        reye_pose_torch = torch.tensor(reye_pose_np[frame_idx], dtype=torch.float32).unsqueeze(0)
        expr_torch = torch.tensor(expr_np[frame_idx], dtype=torch.float32).unsqueeze(0)
        global_orient_torch = torch.tensor(global_orient_np[frame_idx], dtype=torch.float32).unsqueeze(0)
        transl_torch = torch.tensor(transl, dtype=torch.float32).unsqueeze(0)

        # Run the SMPL-X forward pass to get the vertices
        smplx_output = smplx_model(
            betas=betas_torch,
            global_orient=global_orient_torch,
            body_pose=body_pose_torch,
            left_hand_pose=lhand_pose_torch,
            right_hand_pose=rhand_pose_torch,
            jaw_pose=jaw_pose_torch,
            leye_pose=leye_pose_torch,
            reye_pose=reye_pose_torch,
            expression=expr_torch,
            transl=transl_torch
        )

        vertices = smplx_output.vertices[0].detach().cpu().numpy()  # shape (10475, 3) for SMPL-X
        
        # Translate the vertices to origin
        vertices -= vertices.mean(axis=0)
        
        # 4. Export to OBJ
        obj_filename = os.path.join(OUTPUT_FOLDER, f"frame_{frame_idx:04d}.obj")
        with open(obj_filename, "w") as f_obj:
            # Write vertices
            for v in vertices:
                f_obj.write(f"v {v[0]} {v[1]} {v[2]}\n")
            # Write faces (OBJ is 1-based indexing)
            for face in faces:
                f_obj.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
                
        print(f"Saved: {obj_filename}")

if __name__ == "__main__":
    main()
