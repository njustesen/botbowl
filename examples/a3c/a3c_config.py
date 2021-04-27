
# Environment
env_name = "FFAI-3-v3"
pathfinding_enabled = True

# Training configuration
max_updates = 10000
num_processes = 4

# Optimizer config
learning_rate = 0.001
gamma = 0.99
entropy_coef = 0.01
value_loss_coef = 0.5
prediction_loss_coeff = 0.1
max_grad_norm = 0.05

# Architecture
num_hidden_nodes = 128
num_cnn_kernels = [32, 64]

model_name = env_name
log_filename = "logs/" + model_name + ".dat"