
# Environment
env_name = "FFAI-1-v3"
pathfinding_enabled = True

# Training configuration
dict_worker_memory_size = {"FFAI-1-v3": 20,
                           "FFAI-3-v3": 30,
                           "FFAI-5-v3": 300,
                           "FFAI-7-v3": 400}

dict_min_batch_size = {"FFAI-1-v3": 160,
                       "FFAI-3-v3": 350,
                       "FFAI-5-v3": 500,
                       "FFAI-7-v3": 500}

dict_max_steps = {"FFAI-1-v3": 1000000,
                  "FFAI-3-v3": 10000000,
                  "FFAI-5-v3": 100000000,
                  "FFAI-7-v3": 100000000}

max_steps = dict_max_steps[env_name]
num_processes = 8
worker_memory_size = dict_worker_memory_size[env_name]
min_batch_size = dict_min_batch_size[env_name]



# Optimizer config
learning_rate = 0.001
gamma = 0.98
entropy_coef = 0.01
value_loss_coef = 0.5
prediction_loss_coeff = 0.1
max_grad_norm = 0.05

# Architecture
num_hidden_nodes = 128
num_cnn_kernels = [32, 64]

model_name = env_name
log_filename = "logs/" + model_name + ".dat"