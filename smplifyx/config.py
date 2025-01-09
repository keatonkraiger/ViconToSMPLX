# OPTIMIZE_SHAPE={
#     'shape3d':1.0,
#     'reg_shape':5e-3
# }

OPTIMIZE_RT={
    'k3d':1.0,
    'smooth_body':5e-1,
    'smooth_pose':1e-1,
    'reg_pose':1e-2
}

# OPTIMIZE_POSES={
#     'k3d':1,
#     'smooth_body':5e-1,
#     'smooth_pose':1e-1,
#     'reg_pose':1e-2
# }

# OPTIMIZE_HAND={
#     'k3d':1,
#     'smooth_body':5,
#     'smooth_pose':1e-1,
#     'smooth_hand':1e-3,
#     'reg_pose':1e-2,
#     'k3d_hand':10, # 10
#     'reg_hand':1e-4

# }

# OPTIMIZE_EXPR={
#     'k3d':1,
#     'smooth_body':5e-1,
#     'smooth_pose':1e-1,
#     'smooth_hand':1e-3,
#     'smooth_head':1e-3,
#     'reg_pose':1e-2,
#     'k3d_hand':10,
#     'reg_hand':1e-4,
#     'k3d_face':10,
#     'reg_head':1e-2,
#     'reg_expr':1e-2

# }

# Optimization stages and weights adjusted for Vicon marker data

OPTIMIZE_SHAPE={
    'shape3d':1.0,
    'reg_shape':5e-3
}

# Stage 1: Initial pose estimation
OPTIMIZE_INIT = {
    'k3d': 1.0,              # 3D joint position fitting
    'reg_pose': 0.5,         # Pose regularization (reduced from original)
    'smooth_body': 1.0,      # Increased temporal smoothness
    'smooth_pose': 0.2       # Pose smoothness
}

# Stage 2: Body pose refinement
OPTIMIZE_POSES = {
    'k3d': 1.0,
    'smooth_body': 1.0,      # Maintain temporal consistency
    'smooth_pose': 0.2,
    'reg_pose': 0.3         # Slightly relaxed pose regularization
}

# Stage 3: Hand pose optimization
OPTIMIZE_HAND = {
    'k3d': 1.0,
    'smooth_body': 2.0,      # Increased body smoothness during hand optimization
    'smooth_pose': 0.2,
    'smooth_hand': 0.1,      # Gentle hand motion smoothing
    'reg_pose': 0.3,
    'k3d_hand': 5.0,        # Reduced from original to prevent over-fitting
    'reg_hand': 1e-3        # Slight hand pose regularization
}

# Stage 4: Fine details (expression, etc.)
OPTIMIZE_EXPR = {
    'k3d': 1.0,
    'smooth_body': 1.0,
    'smooth_pose': 0.2,
    'smooth_hand': 0.1,
    'smooth_head': 0.1,
    'reg_pose': 0.3,
    'k3d_hand': 5.0,
    'reg_hand': 1e-3,
    'k3d_face': 5.0,        # Reduced face fitting weight
    'reg_head': 0.05,
    'reg_expr': 0.05        # Reduced expression regularization
}