import numpy as np
import torch
import torch.nn as nn

# Define the Vicon marker names in order matching the data
VICON_MARKERS = [
    'LFHD', 'RFHD', 'LBHD', 'RBHD',  # Head
    'C7', 'T10', 'CLAV', 'STRN',      # Torso
    'RBAK', 'LSHO', 'LUPA', 'LELB',   # Left arm start
    'LFRM', 'LWRA', 'LWRB', 'LFIN',   
    'RSHO', 'RUPA', 'RELB', 'RFRM',   # Right arm start
    'RWRA', 'RWRB', 'RFIN', 'LASI',   
    'RASI', 'LPSI', 'RPSI', 'LTHI',   # Pelvis and legs
    'LKNE', 'LTIB', 'LANK', 'LHEE',
    'LTOE', 'RTHI', 'RKNE', 'RTIB',
    'RANK', 'RHEE', 'RTOE'
]

# Create marker index lookup
MARKER_IDX = {name: idx for idx, name in enumerate(VICON_MARKERS)}

# Define joint estimation rules using marker indices
JOINT_ESTIMATION_RULES = {
    'pelvis': {
        'markers': ['LASI', 'RASI', 'LPSI', 'RPSI'],
        'weights': [0.25, 0.25, 0.25, 0.25]  # Equal weights for center
    },
    'left_hip': {
        'markers': ['LASI', 'LTHI'],
        'weights': [0.7, 0.3]  # More weight to ASIS
    },
    'right_hip': {
        'markers': ['RASI', 'RTHI'],
        'weights': [0.7, 0.3]
    },
    'left_knee': {
        'markers': ['LKNE'],
        'weights': [1.0]
    },
    'right_knee': {
        'markers': ['RKNE'],
        'weights': [1.0]
    },
    'left_ankle': {
        'markers': ['LANK'],
        'weights': [1.0]
    },
    'right_ankle': {
        'markers': ['RANK'],
        'weights': [1.0]
    },
    'spine': {
        'markers': ['T10', 'STRN'],
        'weights': [0.5, 0.5]
    },
    'neck': {
        'markers': ['C7', 'CLAV'],
        'weights': [0.6, 0.4]  # More weight to C7
    },
    'head': {
        'markers': ['LFHD', 'RFHD', 'LBHD', 'RBHD'],
        'weights': [0.25, 0.25, 0.25, 0.25]
    },
    'left_shoulder': {
        'markers': ['LSHO'],
        'weights': [1.0]
    },
    'right_shoulder': {
        'markers': ['RSHO'],
        'weights': [1.0]
    },
    'left_elbow': {
        'markers': ['LELB'],
        'weights': [1.0]
    },
    'right_elbow': {
        'markers': ['RELB'],
        'weights': [1.0]
    },
    'left_wrist': {
        'markers': ['LWRA', 'LWRB'],
        'weights': [0.5, 0.5]
    },
    'right_wrist': {
        'markers': ['RWRA', 'RWRB'],
        'weights': [0.5, 0.5]
    }
}

# Define mapping from our estimated joints to SMPL-X joint indices
# Note: These indices need to be verified against SMPL-X documentation
ESTIMATED_TO_SMPLX = {
    'pelvis': 0,
    'left_hip': 1,
    'right_hip': 2,
    'left_knee': 4,
    'right_knee': 5,
    'left_ankle': 7,
    'right_ankle': 8,
    'spine': 3,
    'neck': 12,
    'head': 15,
    'left_shoulder': 13,
    'right_shoulder': 14,
    'left_elbow': 16,
    'right_elbow': 17,
    'left_wrist': 18,
    'right_wrist': 19
}

class ViconJointRegressor(nn.Module):
    def __init__(self):
        super(ViconJointRegressor, self).__init__()
        self.joint_rules = JOINT_ESTIMATION_RULES
        
    def _estimate_joint_position(self, markers, joint_name):
        """
        Estimate single joint position from relevant markers
        Args:
            markers: (batch_size, n_markers, 3) tensor of marker positions
            joint_name: string, key in JOINT_ESTIMATION_RULES
        Returns:
            (batch_size, 3) tensor of estimated joint position
        """
        rule = self.joint_rules[joint_name]
        marker_indices = [MARKER_IDX[m] for m in rule['markers']]
        weights = torch.tensor(rule['weights'], device=markers.device).view(1, -1, 1)
        
        # Get relevant markers and apply weights
        relevant_markers = markers[:, marker_indices]
        joint_position = (relevant_markers * weights).sum(dim=1)
        
        return joint_position
    
    def forward(self, marker_positions):
        """
        Convert marker positions to joint positions
        Args:
            marker_positions: (batch_size, n_markers, 3) tensor
        Returns:
            joint_positions: (batch_size, n_joints, 3) tensor
            confidence: (batch_size, n_joints) tensor
        """
        batch_size = marker_positions.shape[0]
        device = marker_positions.device
        
        # Initialize output tensors
        n_joints = len(ESTIMATED_TO_SMPLX)
        joint_positions = torch.zeros((batch_size, n_joints, 3), device=device)
        confidence = torch.ones((batch_size, n_joints), device=device)
        
        # Estimate each joint position
        for idx, (joint_name, _) in enumerate(ESTIMATED_TO_SMPLX.items()):
            joint_pos = self._estimate_joint_position(marker_positions, joint_name)
            joint_positions[:, idx] = joint_pos
            
            # Calculate confidence based on marker visibility
            rule = self.joint_rules[joint_name]
            marker_indices = [MARKER_IDX[m] for m in rule['markers']]
            # If any marker is missing (contains NaN), reduce confidence
            marker_valid = ~torch.isnan(marker_positions[:, marker_indices]).any(dim=-1)
            confidence[:, idx] = marker_valid.float().mean(dim=-1)
        
        return joint_positions, confidence

def create_joint_regressor():
    """Factory function to create and initialize the joint regressor"""
    return ViconJointRegressor()

# Define indices for body and hand joints based on our estimated joint order
# These correspond to the indices in ESTIMATED_TO_SMPLX mapping

# Get ordered list of joints as they appear in our estimation
JOINT_NAMES = list(ESTIMATED_TO_SMPLX.keys())

# Create indices for body joints (main body + head, no hands)
BODY_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine', 'left_knee', 
    'right_knee', 'left_ankle', 'right_ankle', 'neck', 'head',
    'left_shoulder', 'right_shoulder'
]

# Create indices for hand joints
HAND_JOINT_NAMES = [
    'left_elbow', 'left_wrist',
    'right_elbow', 'right_wrist'
]

# Convert to indices
VICON_BODY = np.array([JOINT_NAMES.index(name) for name in BODY_JOINT_NAMES])
VICON_HAND = np.array([JOINT_NAMES.index(name) for name in HAND_JOINT_NAMES])

# Define joint connections for visualization
VICON_JOINT_LIMBS = [
    # Torso and head connections
    [JOINT_NAMES.index('pelvis'), JOINT_NAMES.index('spine')],
    [JOINT_NAMES.index('spine'), JOINT_NAMES.index('neck')],
    [JOINT_NAMES.index('neck'), JOINT_NAMES.index('head')],
    
    # Left arm connections
    [JOINT_NAMES.index('neck'), JOINT_NAMES.index('left_shoulder')],
    [JOINT_NAMES.index('left_shoulder'), JOINT_NAMES.index('left_elbow')],
    [JOINT_NAMES.index('left_elbow'), JOINT_NAMES.index('left_wrist')],
    
    # Right arm connections
    [JOINT_NAMES.index('neck'), JOINT_NAMES.index('right_shoulder')],
    [JOINT_NAMES.index('right_shoulder'), JOINT_NAMES.index('right_elbow')],
    [JOINT_NAMES.index('right_elbow'), JOINT_NAMES.index('right_wrist')],
    
    # Left leg connections
    [JOINT_NAMES.index('pelvis'), JOINT_NAMES.index('left_hip')],
    [JOINT_NAMES.index('left_hip'), JOINT_NAMES.index('left_knee')],
    [JOINT_NAMES.index('left_knee'), JOINT_NAMES.index('left_ankle')],
    
    # Right leg connections
    [JOINT_NAMES.index('pelvis'), JOINT_NAMES.index('right_hip')],
    [JOINT_NAMES.index('right_hip'), JOINT_NAMES.index('right_knee')],
    [JOINT_NAMES.index('right_knee'), JOINT_NAMES.index('right_ankle')]
]

# For visualization of marker connections (optional, if you want to visualize raw marker data)
VICON_MARKER_LIMBS = [
    # Head
    [MARKER_IDX['LFHD'], MARKER_IDX['RFHD']], 
    [MARKER_IDX['LBHD'], MARKER_IDX['RBHD']], 
    [MARKER_IDX['LFHD'], MARKER_IDX['LBHD']], 
    [MARKER_IDX['RFHD'], MARKER_IDX['RBHD']],
    
    # Torso
    [MARKER_IDX['C7'], MARKER_IDX['CLAV']], 
    [MARKER_IDX['CLAV'], MARKER_IDX['STRN']], 
    [MARKER_IDX['C7'], MARKER_IDX['T10']], 
    [MARKER_IDX['T10'], MARKER_IDX['STRN']],
    
    # Left arm
    [MARKER_IDX['LSHO'], MARKER_IDX['LUPA']], 
    [MARKER_IDX['LUPA'], MARKER_IDX['LELB']], 
    [MARKER_IDX['LELB'], MARKER_IDX['LFRM']], 
    [MARKER_IDX['LFRM'], MARKER_IDX['LWRA']],
    [MARKER_IDX['LFRM'], MARKER_IDX['LWRB']],
    
    # Right arm
    [MARKER_IDX['RSHO'], MARKER_IDX['RUPA']], 
    [MARKER_IDX['RUPA'], MARKER_IDX['RELB']], 
    [MARKER_IDX['RELB'], MARKER_IDX['RFRM']], 
    [MARKER_IDX['RFRM'], MARKER_IDX['RWRA']],
    [MARKER_IDX['RFRM'], MARKER_IDX['RWRB']],
    
    # Pelvis
    [MARKER_IDX['LASI'], MARKER_IDX['RASI']], 
    [MARKER_IDX['LPSI'], MARKER_IDX['RPSI']], 
    [MARKER_IDX['LASI'], MARKER_IDX['LPSI']], 
    [MARKER_IDX['RASI'], MARKER_IDX['RPSI']],
    
    # Left leg
    [MARKER_IDX['LTHI'], MARKER_IDX['LKNE']], 
    [MARKER_IDX['LKNE'], MARKER_IDX['LTIB']], 
    [MARKER_IDX['LTIB'], MARKER_IDX['LANK']], 
    
    # Right leg
    [MARKER_IDX['RTHI'], MARKER_IDX['RKNE']], 
    [MARKER_IDX['RKNE'], MARKER_IDX['RTIB']], 
    [MARKER_IDX['RTIB'], MARKER_IDX['RANK']]
]

if __name__ == '__main__':
    # Test code
    print("Body joint indices:", VICON_BODY)
    print("Hand joint indices:", VICON_HAND)
    
    regressor = create_joint_regressor()
    # Create dummy data: 2 frames, 39 markers, 3 coordinates
    dummy_markers = torch.rand(2, len(VICON_MARKERS), 3)
    joints, conf = regressor(dummy_markers)
    print(f"Input shape: {dummy_markers.shape}")
    print(f"Output joints shape: {joints.shape}")
    print(f"Output confidence shape: {conf.shape}")