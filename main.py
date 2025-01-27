"""
Integration of Vicon marker data with SMPLify-X optimization pipeline.
Processes single mocap files and handles marker-to-SMPLX conversion.
"""
from utils.rotation_conversion import aa2rot_torch
import argparse
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from scipy.io import loadmat
from smplx import SMPLXLayer
#from mmhuman3d.core.visualization import visualize_smpl_pose

from smplifyx.optimize import multi_stage_optimize
from smplifyx.loss import *
from utils.io import write_smplx, save_as_obj
from utils.torch_utils import init_params
#from utils.visualize_smplx import visualize_smplx_model
from utils.vicon_mapping import ViconJointRegressor, create_joint_regressor

def load_mocap_data(mocap_file):
    """Load mocap data from .mat file"""
    data = loadmat(mocap_file)
    # Shape: (N, 39, 4) where last dim is [x,y,z,conf]
    markers = data['mcMarkers']
    
    # Ensure correct shape and handle confidence
    assert markers.shape[1] == 39, f"Expected 39 markers, got {markers.shape[1]}"
    
    markers_tensor = torch.tensor(markers, dtype=torch.float32)
    return markers_tensor

def process_markers_to_joints(markers, joint_regressor, device='cuda'):
    """Convert marker positions to joint positions with confidence"""
    # Move to device
    markers = markers.to(device)
    
    # Split positions and confidence
    positions = markers[..., :3]
    confidence = markers[..., 3]
    
    # Get joint positions and joint confidence from regressor
    joint_positions, joint_conf = joint_regressor(positions)
    
    # Scale joint confidence by marker confidence
    # Properly reshape for broadcasting
    # confidence shape: [frames, markers] -> [frames, 1]
    marker_conf_scale = confidence.mean(dim=-1, keepdim=True)  # [frames, 1]
    # joint_conf shape: [frames, n_joints]
    final_conf = joint_conf * marker_conf_scale  # [frames, n_joints]
    
    # Combine into format expected by SMPLify-X
    # joint_positions: [frames, n_joints, 3]
    # final_conf: [frames, n_joints] -> [frames, n_joints, 1]
    joints_with_conf = torch.cat([
        joint_positions,
        final_conf.unsqueeze(-1)
    ], dim=-1)  # Result: [frames, n_joints, 4]
    
    return joints_with_conf

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load mocap data
    print(f'Loading mocap data from: {args.mocap_file}')
    marker_data = load_mocap_data(args.mocap_file)
    # Take every Nth frame
    marker_data = marker_data
    num_frames = marker_data.shape[0]
    print(f'Loaded {num_frames} frames of motion data')
    
    # Create joint regressor
    joint_regressor = create_joint_regressor().to(device)
    
    # Process markers to get joint positions
    print('Converting markers to joint positions...')
    joint_positions = process_markers_to_joints(marker_data, joint_regressor, device)
    
    # Initialize SMPL-X body model
    print(f'Initializing SMPL-X model with gender: {args.gender}')
    body_model = SMPLXLayer(
        model_path=args.model_folder,
        gender=args.gender,
        num_betas=10,
        use_pca=False,
        flat_hand_mean=True,
    ).to(device)

    # Initialize parameters
    params = {
        'body_pose': np.zeros((num_frames, 21, 3)),
        'global_orient': np.zeros((num_frames, 3)),
        'transl': np.zeros((num_frames, 3)),
        'lhand_pose': np.zeros((num_frames, 15, 3)),
        'rhand_pose': np.zeros((num_frames, 15, 3)),
        'jaw_pose': np.zeros((num_frames, 3)),
        'leye_pose': np.zeros((num_frames, 3)),
        'reye_pose': np.zeros((num_frames, 3)),
        'expression': np.zeros((num_frames, 10)),
        'betas': np.zeros((num_frames, 10)),  # Shape parameters for each frame
    }
    params = init_params(params, body_model, num_frames)

    print('Starting SMPLify-X optimization...')
    start_time = time.time()
    
    # Run optimization
    optimized_params = multi_stage_optimize(
        params=params,
        body_models=body_model,
        kp3ds=joint_positions
    )
    
    end_time = time.time()
    print(f'Optimization completed in {end_time - start_time:.2f} seconds')

    if args.output_file:
        write_smplx(optimized_params, args.output_file)
        print(f'Results saved to: {args.output_file}')
         
    if args.save_obj:
        smplx_model = SMPLXLayer(model_path=args.model_folder, gender='neutral')
        global_orient_mat = aa2rot_torch(params['global_orient'])     
        global_orient_mat = global_orient_mat.unsqueeze(1)  # [100, 1, 3, 3]
        body_pose = params['body_pose']
        body_pose_mat = aa2rot_torch(body_pose.reshape(-1, 3)).reshape(-1, 21, 3, 3)  # [100, 21, 3, 3] 
        transl = torch.tensor(params['transl'])  # [100, 3]
        betas = torch.tensor(params['betas'])  # [100, 10]
        # move each mat to cpu
        global_orient_mat = global_orient_mat.cpu()
        body_pose_mat = body_pose_mat.cpu()
        transl = transl.cpu()
        betas = betas.cpu()
    
        output = smplx_model(
            betas=betas,
            global_orient=global_orient_mat,
            body_pose=body_pose_mat,
            transl=transl
            )
    
        vertices = output.vertices.cpu().numpy()
        faces = smplx_model.faces  # [F, 3]
        output_file = "smplx_output_frame_0.obj"
        breakpoint()
        save_as_obj(output_file, vertices[0], faces)
        


    # Visualize if requested
    '''
    if args.visualize:
        print('Visualizing results...')
        visualize_smplx_model(
            optimized_params,
            args.gender,
            joint_positions if args.vis_joints else None,
            args.vis_joints
        )'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process mocap data through SMPLify-X')
    
    parser.add_argument('--mocap_file', type=str, required=True,
                        help='Path to mocap .mat file')
    parser.add_argument('--model_folder', type=str, default='./body_models/smplx',
                        help='Path to SMPL-X model folder')
    parser.add_argument('--gender', type=str, default='neutral',
                        choices=['neutral', 'male', 'female'],
                        help='Gender for SMPL-X model')
    parser.add_argument('--output_file', type=str,
                        help='Path to save SMPL-X parameters')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results')
    parser.add_argument('--vis_joints', action='store_true',
                        help='Visualize estimated joint positions')
    parser.add_argument('--save_obj', action='store_false',
                        help='Save results as .obj files')    
    args = parser.parse_args()
    main(args)
