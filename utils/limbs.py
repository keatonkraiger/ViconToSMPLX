"""
Defines the marker and limb configurations for Vicon data.
This follows a similar structure to the original limbs.py but adapted for Vicon markers.
"""

import numpy as np

# Define the marker connections for visualization
VICON_MARKER_LIMBS = [
    # Head
    ['LFHD', 'RFHD'], ['LBHD', 'RBHD'], ['LFHD', 'LBHD'], ['RFHD', 'RBHD'],
    
    # Torso
    ['C7', 'CLAV'], ['CLAV', 'STRN'], ['C7', 'T10'], ['T10', 'STRN'],
    ['RBAK', 'T10'], ['RBAK', 'CLAV'],
    
    # Left Arm
    ['LSHO', 'CLAV'], ['LSHO', 'LUPA'], ['LUPA', 'LELB'],
    ['LELB', 'LFRM'], ['LFRM', 'LWRA'], ['LFRM', 'LWRB'],
    ['LWRA', 'LFIN'], ['LWRB', 'LFIN'],
    
    # Right Arm
    ['RSHO', 'CLAV'], ['RSHO', 'RUPA'], ['RUPA', 'RELB'],
    ['RELB', 'RFRM'], ['RFRM', 'RWRA'], ['RFRM', 'RWRB'],
    ['RWRA', 'RFIN'], ['RWRB', 'RFIN'],
    
    # Pelvis
    ['LASI', 'RASI'], ['LPSI', 'RPSI'], ['LASI', 'LPSI'], ['RASI', 'RPSI'],
    
    # Left Leg
    ['LASI', 'LTHI'], ['LTHI', 'LKNE'], ['LKNE', 'LTIB'],
    ['LTIB', 'LANK'], ['LANK', 'LHEE'], ['LANK', 'LTOE'],
    
    # Right Leg
    ['RASI', 'RTHI'], ['RTHI', 'RKNE'], ['RKNE', 'RTIB'],
    ['RTIB', 'RANK'], ['RANK', 'RHEE'], ['RANK', 'RTOE']
]

# Convert marker names to indices for the visualization code
def create_marker_limb_indices(marker_names):
    """Convert marker name pairs to index pairs"""
    marker_to_idx = {name: idx for idx, name in enumerate(marker_names)}
    return [[marker_to_idx[start], marker_to_idx[end]] 
            for start, end in VICON_MARKER_LIMBS]

# Define estimated joint connections for visualization
VICON_JOINT_LIMBS = [
    # Torso
    ['pelvis', 'spine'], ['spine', 'neck'], ['neck', 'head'],
    
    # Left Arm
    ['neck', 'left_shoulder'], ['left_shoulder', 'left_elbow'],
    ['left_elbow', 'left_wrist'],
    
    # Right Arm
    ['neck', 'right_shoulder'], ['right_shoulder', 'right_elbow'],
    ['right_elbow', 'right_wrist'],
    
    # Left Leg
    ['pelvis', 'left_hip'], ['left_hip', 'left_knee'],
    ['left_knee', 'left_ankle'],
    
    # Right Leg
    ['pelvis', 'right_hip'], ['right_hip', 'right_knee'],
    ['right_knee', 'right_ankle']
]

def create_joint_limb_indices(joint_names):
    """Convert joint name pairs to index pairs"""
    joint_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    return [[joint_to_idx[start], joint_to_idx[end]] 
            for start, end in VICON_JOINT_LIMBS]

# Group markers by body segment for joint estimation
MARKER_GROUPS = {
    'HEAD': ['LFHD', 'RFHD', 'LBHD', 'RBHD'],
    'TORSO': ['C7', 'T10', 'CLAV', 'STRN', 'RBAK'],
    'PELVIS': ['LASI', 'RASI', 'LPSI', 'RPSI'],
    'LEFT_ARM': ['LSHO', 'LUPA', 'LELB', 'LFRM', 'LWRA', 'LWRB', 'LFIN'],
    'RIGHT_ARM': ['RSHO', 'RUPA', 'RELB', 'RFRM', 'RWRA', 'RWRB', 'RFIN'],
    'LEFT_LEG': ['LTHI', 'LKNE', 'LTIB', 'LANK', 'LHEE', 'LTOE'],
    'RIGHT_LEG': ['RTHI', 'RKNE', 'RTIB', 'RANK', 'RHEE', 'RTOE']
}

# Marker weights for joint estimation
MARKER_WEIGHTS = {
    'PELVIS': {
        'LASI': 0.25, 'RASI': 0.25, 'LPSI': 0.25, 'RPSI': 0.25
    },
    'SPINE': {
        'T10': 0.5, 'STRN': 0.5
    },
    'NECK': {
        'C7': 0.6, 'CLAV': 0.4
    },
    'HEAD': {
        'LFHD': 0.25, 'RFHD': 0.25, 'LBHD': 0.25, 'RBHD': 0.25
    }
}

if __name__ == '__main__':
    # Test code to verify limb definitions
    from utils.vicon_mapping import VICON_MARKERS
    
    marker_limbs = create_marker_limb_indices(VICON_MARKERS)
    print("Number of marker connections:", len(marker_limbs))
    
    # Test joint limbs
    joint_names = list(ESTIMATED_TO_SMPLX.keys())
    joint_limbs = create_joint_limb_indices(joint_names)
    print("Number of joint connections:", len(joint_limbs))