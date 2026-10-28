"""Generate flight C code for a trained policy.

Port of optimal_quad_control_RL/generate_c_code.py (the flight-verified
export) driven by a run config instead of hardcoded env parameters. Writes:

    neural_network.c/.h  - the policy network: nn_forward(input, output)
    nn_controller.c/.h   - nn_control(world_state, motor_cmds): builds the
                           network input from the world state (same as the
                           env observation), runs the network and maps the
                           output to motor commands in [0,1]

`verify_c_code` compiles the result and checks it against the Python policy,
open loop on random inputs and closed loop in the simulator.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import tempfile

import numpy as np
import torch
import torch.nn as nn

from airevolve.simulator.simulation.dynamics_params import W_MIN_N

# np.float32 to str
float_to_str = lambda x: str(float(x))
array = lambda values: ', '.join(map(float_to_str, values))


def policy_network(model):
    """Deterministic policy as a torch Sequential (mlp + action layer) and its std."""
    network = nn.Sequential(*(list(model.policy.mlp_extractor.policy_net) + [model.policy.action_net])).cpu()
    network_std = model.policy.log_std.exp().cpu().detach().numpy()
    return network, network_std


def check_supported(model, env, task):
    num_motors = env.action_space.shape[0]
    if model.observation_space.shape[0] != env.state_len or model.action_space.shape[0] != num_motors:
        raise ValueError(
            f'Policy has {model.observation_space.shape[0]} inputs and {model.action_space.shape[0]} outputs, but the '
            f'{task} controller builds {env.state_len} inputs for {num_motors} motors.')
    if env.num_state_history or env.num_action_history:
        raise ValueError('State/action history inputs are not supported by the C controller.')


def motor_table(drone_sim):
    """Comment lines describing each motor, to check the FC motor mapping."""
    lines = ['// Motor order (motor_cmds[i] drives flight-controller motor i+1), body frame FRD:']
    for i, prop in enumerate(drone_sim.config.propellers):
        loc = ', '.join(f'{v:+.3f}' for v in prop['loc'])
        axis = ', '.join(f'{v:+.3f}' for v in drone_sim.params['axis'][i])
        lines.append(f'//   motor {i+1}: position [{loc}] m, thrust axis [{axis}], spin {prop["dir"][3]}')
    return '\n'.join(lines)


def write_network(network, output_folder):
    source_file_path = os.path.join(output_folder, 'neural_network.c')
    header_file_path = os.path.join(output_folder, 'neural_network.h')

    with open(source_file_path, 'w') as file:
        file.write('#include "neural_network.h"\n')
        file.write('#include <math.h>\n\n')

        # Define weights and biases as global constant float arrays
        linear_layers = [layer for layer in network if isinstance(layer, nn.Linear)]
        for i, layer in enumerate(linear_layers, start=1):
            weights_layer = layer.weight.data.cpu().numpy()
            biases_layer = layer.bias.data.cpu().numpy()

            file.write(f'const float weights_fc{i}[] = {{\n')
            file.write(',\n'.join([', '.join(map(float_to_str, row)) for row in weights_layer]))
            file.write('\n};\n\n')

            file.write(f'const float biases_fc{i}[] = {{\n')
            file.write(', '.join(map(float_to_str, biases_layer)))
            file.write('\n};\n\n')

        file.write('''void nn_linear(const float* weights, const float* biases, const float* input, int in_features, int out_features, float* output) {
    for (int i = 0; i < out_features; ++i) {
        float neuron = biases[i];
        for (int j = 0; j < in_features; ++j) {
            neuron += input[j] * weights[i * in_features + j];
        }
        output[i] = neuron;
    }
}

void nn_relu(float* input, int size) {
    for (int i = 0; i < size; ++i) {
        input[i] = fmaxf(0, input[i]);
    }
}

void nn_tanh(float* input, int size) {
    for (int i = 0; i < size; ++i) {
        input[i] = tanhf(input[i]);
    }
}

''')

        # FORWARD FUNCTION
        file.write('void nn_forward(const float* input, float* output) {\n')
        input_array = 'input'
        layer_size = None
        i = 0
        for layer in network:
            if isinstance(layer, nn.Linear):
                i += 1
                if i < len(linear_layers):
                    file.write(f'    float fc{i}_output[{layer.out_features}];\n')
                    file.write(f'    nn_linear(weights_fc{i}, biases_fc{i}, {input_array}, {layer.in_features}, {layer.out_features}, fc{i}_output);\n')
                    input_array = f'fc{i}_output'
                else:
                    file.write(f'    nn_linear(weights_fc{i}, biases_fc{i}, {input_array}, {layer.in_features}, {layer.out_features}, output);\n')
                    input_array = 'output'
                layer_size = layer.out_features
            elif isinstance(layer, nn.ReLU):
                file.write(f'    nn_relu({input_array}, {layer_size});\n')
            elif isinstance(layer, nn.Tanh):
                file.write(f'    nn_tanh({input_array}, {layer_size});\n')
            else:
                raise Exception(f'Unsupported layer: {layer}')
        file.write('}\n')

    with open(header_file_path, 'w') as file:
        file.write('#ifndef NEURAL_NETWORK_H\n')
        file.write('#define NEURAL_NETWORK_H\n\n')
        file.write('void nn_forward(const float* input, float* output);\n')
        file.write('\n#endif // NEURAL_NETWORK_H\n')

    return [source_file_path, header_file_path]


def write_controller(env, task, network_std, output_folder, model_path='', deterministic=False):
    name = 'nn_controller'
    source_file_path = os.path.join(output_folder, f'{name}.c')
    header_file_path = os.path.join(output_folder, f'{name}.h')

    num_motors = env.action_space.shape[0]
    num_states = env.world_states.shape[1]
    nn_input_size = env.state_len
    motor_lim = env.motor_limit
    u_max = 2*motor_lim - 1
    w_min_n = W_MIN_N
    w_max_n = env.drone_sim.w_norm_max

    # task specific declarations for the header
    if task == 'gates':
        task_header = f'''#define GATES_AHEAD {env.gates_ahead}
#define NUM_GATES {env.num_gates}

extern const float gate_pos[NUM_GATES][3];
extern const float gate_yaw[NUM_GATES];
extern const float start_pos[3];
extern const float start_yaw;
extern uint8_t target_gate_index;
'''
    else:
        task_header = '''extern const float target_pos[3];
'''

    with open(header_file_path, 'w') as file:
        file.write(f'''#ifndef {name.upper()}_H
#define {name.upper()}_H

#include <stdint.h>
#include <stdbool.h>

// Include the neural network code
#include "neural_network.h"

// Policy trained on the {task} task: {model_path}
{motor_table(env.drone_sim)}
#define NUM_MOTORS {num_motors}
#define NUM_STATES {num_states}
#define NN_INPUT_SIZE {nn_input_size}
{task_header}
// if false, gaussian exploration noise is added to the network output
extern bool deterministic;

// resets the controller at the start of a flight
void nn_reset(void);
// world_state: x y z vx vy vz phi theta psi p q r w1..w{num_motors} (NED position/velocity, euler angles,
// body rates, motor speeds in rad/s); motor_cmds: normalized motor commands in [0,1]
void nn_control(const float world_state[NUM_STATES], float motor_cmds[NUM_MOTORS]);

#endif
''')

    # task specific definitions and network input
    if task == 'gates':
        task_definitions = f'''// gate positions and headings
const float gate_pos[NUM_GATES][3] = {{
{''.join(f'    {{{array(p)}}},{chr(10)}' for p in env.gate_pos)}}};

const float gate_yaw[NUM_GATES] = {{
    {array(env.gate_yaw)}
}};

const float start_pos[3] = {{{array(env.start_pos)}}};
const float start_yaw = {float_to_str(env.gate_yaw[0])};

// position and heading of each gate in the frame of the previous gate
const float gate_pos_rel[NUM_GATES][3] = {{
{''.join(f'    {{{array(p)}}},{chr(10)}' for p in env.gate_pos_rel)}}};

const float gate_yaw_rel[NUM_GATES] = {{
    {array(env.gate_yaw_rel)}
}};

uint8_t target_gate_index = 0;

void nn_reset(void) {{
    target_gate_index = 0;
}}
'''
        task_input = '''    // Get the position and heading of the target gate
    float target_pos[3] = {gate_pos[target_gate_index][0], gate_pos[target_gate_index][1], gate_pos[target_gate_index][2]};
    float target_yaw = gate_yaw[target_gate_index];

    // Set the target gate index to the next gate if we passed through the current one
    if (cosf(target_yaw) * (pos[0] - target_pos[0]) + sinf(target_yaw) * (pos[1] - target_pos[1]) > 0) {
        target_gate_index++;
        // loop back to the first gate if we reach the end
        target_gate_index = target_gate_index % NUM_GATES;
        // reset the target position and heading
        target_pos[0] = gate_pos[target_gate_index][0];
        target_pos[1] = gate_pos[target_gate_index][1];
        target_pos[2] = gate_pos[target_gate_index][2];
        target_yaw = gate_yaw[target_gate_index];
    }

    // Get the position of the drone in gate frame
    float pos_rel[3] = {
        cosf(target_yaw) * (pos[0] - target_pos[0]) + sinf(target_yaw) * (pos[1] - target_pos[1]),
        -sinf(target_yaw) * (pos[0] - target_pos[0]) + cosf(target_yaw) * (pos[1] - target_pos[1]),
        pos[2] - target_pos[2]
    };

    // Get the velocity of the drone in gate frame
    float vel_rel[3] = {
        cosf(target_yaw) * vel[0] + sinf(target_yaw) * vel[1],
        -sinf(target_yaw) * vel[0] + cosf(target_yaw) * vel[1],
        vel[2]
    };

    // Get the heading of the drone in gate frame
    float yaw_rel = yaw - target_yaw;
'''
        task_input_extra = '''
    // relative gate positions and headings
    for (int i = 0; i < GATES_AHEAD; i++) {
        uint8_t index = target_gate_index + i + 1;
        // loop back to the first gate if we reach the end
        index = index % NUM_GATES;
        nn_input[12+NUM_MOTORS+4*i]   = gate_pos_rel[index][0];
        nn_input[12+NUM_MOTORS+4*i+1] = gate_pos_rel[index][1];
        nn_input[12+NUM_MOTORS+4*i+2] = gate_pos_rel[index][2];
        nn_input[12+NUM_MOTORS+4*i+3] = gate_yaw_rel[index];
    }
'''
    else:
        task_definitions = f'''// hover target position
const float target_pos[3] = {{{array(env.target_pos)}}};

void nn_reset(void) {{
}}
'''
        task_input = '''    // Get the position of the drone relative to the hover target
    float pos_rel[3] = {pos[0] - target_pos[0], pos[1] - target_pos[1], pos[2] - target_pos[2]};
    float vel_rel[3] = {vel[0], vel[1], vel[2]};
    float yaw_rel = yaw;
'''
        task_input_extra = ''

    with open(source_file_path, 'w') as file:
        file.write(f'''#include "{name}.h"
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

bool deterministic = {'true' if deterministic else 'false'};

// standard deviation of the gaussian policy
const float output_std[NUM_MOTORS] = {{
    {array(network_std)}
}};

{task_definitions}
void nn_control(const float world_state[NUM_STATES], float motor_cmds[NUM_MOTORS]) {{
    // Get the current position, velocity and heading
    float pos[3] = {{world_state[0], world_state[1], world_state[2]}};
    float vel[3] = {{world_state[3], world_state[4], world_state[5]}};
    float yaw = world_state[8];

{task_input}    while (yaw_rel > M_PI) {{yaw_rel -= 2*M_PI;}}
    while (yaw_rel < -M_PI) {{yaw_rel += 2*M_PI;}}

    // Get the neural network input
    float nn_input[NN_INPUT_SIZE];
    // position and velocity
    for (int i = 0; i < 3; i++) {{
        nn_input[i] = pos_rel[i];
        nn_input[i+3] = vel_rel[i];
    }}
    // attitude
    nn_input[6] = world_state[6];
    nn_input[7] = world_state[7];
    nn_input[8] = yaw_rel;
    // body rates
    nn_input[9] = world_state[9];
    nn_input[10] = world_state[10];
    nn_input[11] = world_state[11];
    // motor speeds scaled to [-1,1]
    float w_min = {float_to_str(w_min_n)};
    float w_max = {float_to_str(w_max_n)};
    for (int i = 0; i < NUM_MOTORS; i++) {{
        nn_input[12+i] = (world_state[12+i] - w_min) * 2 / (w_max - w_min) - 1;
    }}
{task_input_extra}
    // Get the neural network output
    float nn_output[NUM_MOTORS];
    nn_forward(nn_input, nn_output);

    // add gaussian noise to the output
    if (!deterministic) {{
        for (int i = 0; i < NUM_MOTORS; i++) {{
            // generate random gaussian variables using the Box-Muller transform
            float u1 = ((float)rand() + 1) / ((float)RAND_MAX + 1);
            float u2 = (float)rand() / (float)RAND_MAX;
            float rand_std = sqrtf(-2 * logf(u1)) * cosf(2 * M_PI * u2);
            // add the noise to the output
            nn_output[i] += output_std[i] * rand_std;
        }}
    }}

    for (int i = 0; i < NUM_MOTORS; i++) {{
        // clip the output to the range [-1, {u_max}] ({motor_lim} MOTOR LIMIT)
        if (nn_output[i] > {u_max}) {{nn_output[i] = {u_max};}}
        if (nn_output[i] < -1) {{nn_output[i] = -1;}}
        // map the output from [-1,1] to [0,1]
        motor_cmds[i] = (nn_output[i] + 1) / 2;
    }}
}}
''')

    return [source_file_path, header_file_path]


def generate_c_code(model, env, task, output_folder, model_path='', deterministic=False):
    """Write the network and controller C code; returns the written paths."""
    check_supported(model, env, task)
    os.makedirs(output_folder, exist_ok=True)
    network, network_std = policy_network(model)
    paths = write_network(network, output_folder)
    paths += write_controller(env, task, network_std, output_folder, model_path, deterministic)
    return paths


def verify_c_code(model, env, task, output_folder, num_episodes=10):
    """Compile the generated code and compare it with the Python policy.

    `env` must be a single-env instance. Returns a dict with the open-loop
    network error, the closed-loop controller error and per-episode
    (steps, gates, crashed) tuples.
    """
    network, _ = policy_network(model)
    num_motors = env.action_space.shape[0]
    nn_input_size = env.state_len
    w_min_n, w_max_n = W_MIN_N, env.drone_sim.w_norm_max

    # compile into a shared library outside the output folder
    build_dir = tempfile.mkdtemp()
    lib_path = os.path.join(build_dir, 'libnn.so')
    sources = [os.path.join(output_folder, f) for f in ['neural_network.c', 'nn_controller.c']]
    subprocess.run(['cc', '-O2', '-Wall', '-fPIC', '-shared', '-o', lib_path, *sources, '-lm'], check=True)
    lib = ctypes.CDLL(lib_path)
    lib.nn_forward.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    lib.nn_control.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    ctypes.c_bool.in_dll(lib, 'deterministic').value = True

    def c_forward(x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = (ctypes.c_float*num_motors)()
        lib.nn_forward(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out)
        return np.array(out[:])

    def c_control(world_state):
        x = np.array(world_state, dtype=np.float32)
        # the simulator stores motor speeds normalized to [-1,1], the C controller expects rad/s
        x[12:] = (x[12:] + 1)/2*(w_max_n - w_min_n) + w_min_n
        out = (ctypes.c_float*num_motors)()
        lib.nn_control(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out)
        # map back to [-1,1]
        return np.array(out[:])*2 - 1

    # 1. network: C vs torch on random inputs
    x = np.random.uniform(-2, 2, size=(1000, nn_input_size)).astype(np.float32)
    with torch.no_grad():
        torch_out = network(torch.tensor(x)).numpy()
    net_error = np.abs(np.array([c_forward(xi) for xi in x]) - torch_out).max()

    # 2. controller: fly the C controller in the simulator and compare every action with the Python policy
    env.reset()
    lib.nn_reset()
    control_error = 0.
    episodes = []
    steps = gates = 0
    while len(episodes) < num_episodes:
        py_action, _ = model.predict(env.states, deterministic=True)
        c_action = c_control(env.world_states[0])
        control_error = max(control_error, np.abs(c_action - py_action[0]).max())
        env.step_async(c_action[np.newaxis].astype(np.float32))
        _, _, dones, infos = env.step_wait()
        steps += 1
        gates += bool(infos[0].get('gate_passed', False))
        if dones[0]:
            episodes.append((steps, gates, steps < env.max_steps))
            steps = gates = 0
            lib.nn_reset()

    return {'net_error': float(net_error), 'control_error': float(control_error), 'episodes': episodes}
