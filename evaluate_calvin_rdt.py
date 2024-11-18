# MIT License

# Copyright (c) 2021 Oier Mees
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Code to evaluate Calvin."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
import copy
from moviepy.editor import ImageSequenceClip
from scipy.spatial.transform import Rotation as R
from PIL import Image

# This is for using the locally installed repo clone when using slurm
# from calvin_agent.models.calvin_base_model import CalvinBaseModel
from calvin_model import RoboticDiffusionTransformerModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
)
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

from evaluation.calvin_evaluation import GR1CalvinEvaluation
from utils.calvin_utils import print_and_save

logger = logging.getLogger(__name__)

# Path to calvin
CALVIN_ROOT = os.environ['CALVIN_ROOT']

EP_LEN = 360
NUM_SEQUENCES = 1000

STATE_VEC_IDX_MAPPING = {
    # [0, 10): right arm joint positions
    **{
        'arm_joint_{}_pos'.format(i): i for i in range(10)
    },
    **{
        'right_arm_joint_{}_pos'.format(i): i for i in range(10)
    },
    # [10, 15): right gripper joint positions
    **{
        'gripper_joint_{}_pos'.format(i): i + 10 for i in range(5)
    },
    **{
        'right_gripper_joint_{}_pos'.format(i): i + 10 for i in range(5)
    },
    'gripper_open': 10, # alias of right_gripper_joint_0_pos
    'right_gripper_open': 10,
    # [15, 25): right arm joint velocities
    **{
        'arm_joint_{}_vel'.format(i): i + 15 for i in range(10)
    },
    **{
        'right_arm_joint_{}_vel'.format(i): i + 15 for i in range(10)
    },
    # [25, 30): right gripper joint velocities
    **{
        'gripper_joint_{}_vel'.format(i): i + 25 for i in range(5)
    },
    **{
        'right_gripper_joint_{}_vel'.format(i): i + 25 for i in range(5)
    },
    'gripper_open_vel': 25, # alias of right_gripper_joint_0_vel
    'right_gripper_open_vel': 25,
    # [30, 33): right end effector positions
    'eef_pos_x': 30,
    'right_eef_pos_x': 30,
    'eef_pos_y': 31,
    'right_eef_pos_y': 31,
    'eef_pos_z': 32,
    'right_eef_pos_z': 32,
    # [33, 39): right end effector 6D pose
    'eef_angle_0': 33,
    'right_eef_angle_0': 33,
    'eef_angle_1': 34,
    'right_eef_angle_1': 34,
    'eef_angle_2': 35,
    'right_eef_angle_2': 35,
    'eef_angle_3': 36,
    'right_eef_angle_3': 36,
    'eef_angle_4': 37,
    'right_eef_angle_4': 37,
    'eef_angle_5': 38,
    'right_eef_angle_5': 38,
    # [39, 42): right end effector velocities
    'eef_vel_x': 39,
    'right_eef_vel_x': 39,
    'eef_vel_y': 40,
    'right_eef_vel_y': 40,
    'eef_vel_z': 41,
    'right_eef_vel_z': 41,
    # [42, 45): right end effector angular velocities
    'eef_angular_vel_roll': 42,
    'right_eef_angular_vel_roll': 42,
    'eef_angular_vel_pitch': 43,
    'right_eef_angular_vel_pitch': 43,
    'eef_angular_vel_yaw': 44,
    'right_eef_angular_vel_yaw': 44,
    # [45, 50): reserved
    # [50, 60): left arm joint positions
    **{
        'left_arm_joint_{}_pos'.format(i): i + 50 for i in range(10)
    },
    # [60, 65): left gripper joint positions
    **{
        'left_gripper_joint_{}_pos'.format(i): i + 60 for i in range(5)
    },
    'left_gripper_open': 60, # alias of left_gripper_joint_0_pos
    # [65, 75): left arm joint velocities
    **{
        'left_arm_joint_{}_vel'.format(i): i + 65 for i in range(10)
    },
    # [75, 80): left gripper joint velocities
    **{
        'left_gripper_joint_{}_vel'.format(i): i + 75 for i in range(5)
    },
    'left_gripper_open_vel': 75, # alias of left_gripper_joint_0_vel
    # [80, 83): left end effector positions
    'left_eef_pos_x': 80,
    'left_eef_pos_y': 81,
    'left_eef_pos_z': 82,
    # [83, 89): left end effector 6D pose
    'left_eef_angle_0': 83,
    'left_eef_angle_1': 84,
    'left_eef_angle_2': 85,
    'left_eef_angle_3': 86,
    'left_eef_angle_4': 87,
    'left_eef_angle_5': 88,
    # [89, 92): left end effector velocities
    'left_eef_vel_x': 89,
    'left_eef_vel_y': 90,
    'left_eef_vel_z': 91,
    # [92, 95): left end effector angular velocities
    'left_eef_angular_vel_roll': 92,
    'left_eef_angular_vel_pitch': 93,
    'left_eef_angular_vel_yaw': 94,
    # [95, 100): reserved
    # [100, 102): base linear velocities
    'base_vel_x': 100,
    'base_vel_y': 101,
    # [102, 103): base angular velocities
    'base_angular_vel': 102,
    # [103, 128): reserved
}
STATE_VEC_LEN = 128


def make_env(dataset_path, observation_space, device_id):
    val_folder = Path(dataset_path) / "validation"
    from evaluation.calvin_env_wrapper_raw import CalvinEnvWrapperRaw
    device = torch.device('cuda', device_id)
    env = CalvinEnvWrapperRaw(val_folder, observation_space, device)
    return env


def evaluate_policy(model, env, eval_sr_path, eval_result_path, eval_dir=None, debug=False):
    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    eval_dir = get_log_dir(eval_dir)
    eval_sequences = get_sequences(NUM_SEQUENCES)

    results = []
    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    sequence_i = 0
    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i)
        results.append(result)
        if not debug:
            success_list = count_success(results)
            average_rate = sum(success_list) / len(success_list) * 5
            with open(eval_sr_path, 'a') as f:
                line =f"{sequence_i}/{NUM_SEQUENCES}: "
                for sr in success_list:
                    line += f"{sr:.3f} | "
                sequence_i += 1
                line += "\n"
                f.write(line)
            description = " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)])
            description += f" Average: {average_rate:.1f} |"
            eval_sequences.set_description(description)
        else:
            sequence_i += 1
    print_and_save(results, eval_sequences, eval_result_path, None)
    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout(env, model, task_checker, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def capitalize_and_period(instr: str) -> str:
    """
    Capitalize the first letter of a string and add a period to the end if it's not there.
    """
    if len(instr) > 0:
        # if the first letter is not capital, make it so
        if not instr[0].isupper():
            # if the first letter is not capital, make it so
            instr = instr[0].upper() + instr[1:]
        # add period to the end if it's not there
        if instr[-1] != '.':
            # add period to the end if it's not there
            instr = instr + '.'
    return instr


"""
Below is a continuous 6D rotation representation adapted from
On the Continuity of Rotation Representations in Neural Networks
https://arxiv.org/pdf/1812.07035.pdf
https://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tools.py
"""
def rotation_matrix_to_ortho6d(matrix: np.ndarray) -> np.ndarray:
    """
    The orhto6d represents the first two column vectors a1 and a2 of the
    rotation matrix: [ | , |,  | ]
                     [ a1, a2, a3]
                     [ | , |,  | ]
    Input: (A1, ..., An, 3, 3)
    Output: (A1, ..., An, 6)
    """
    ortho6d = matrix[..., :, :2]
    # Transpose the last two dimension
    perm = list(range(len(ortho6d.shape)))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    ortho6d = np.transpose(ortho6d, perm)
    # Flatten the last two dimension
    ortho6d = ortho6d.reshape(ortho6d.shape[:-2] + (-1,))
    return ortho6d


def normalize_vector(v: np.ndarray) -> np.ndarray:
    v_mag = np.linalg.norm(v, ord=2, axis=-1, keepdims=True)
    v_mag = np.maximum(v_mag, 1e-8)
    return v / v_mag


def ortho6d_to_rotation_matrix(ortho6d: np.ndarray) -> np.ndarray:
    """
    The orhto6d represents the first two column vectors a1 and a2 of the
    rotation matrix: [ | , |,  | ]
                     [ a1, a2, a3]
                     [ | , |,  | ]
    Input: (A1, ..., An, 6)
    Output: (A1, ..., An, 3, 3)
    """
    x_raw = ortho6d[..., 0:3]
    y_raw = ortho6d[..., 3:6]

    x = normalize_vector(x_raw)
    z = np.cross(x, y_raw)
    z = normalize_vector(z)
    y = np.cross(z, x)

    matrix = np.stack([x, y, z], axis=-1)
    return matrix


def robot_obs_to_state_vec(robot_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eef_pos = robot_obs[:3]
    eef_ang = R.from_euler("xyz", robot_obs[3:6]).as_matrix()
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    gripper_open = (robot_obs[14] + 1) / 2
    gripper_open = np.array([gripper_open])
    qpos = robot_obs[7:14]

    arm_concat = np.concatenate([qpos, gripper_open, eef_pos, eef_ang])
    arm_format = "arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,arm_joint_6_pos,gripper_open,eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5"

    state_vec = np.zeros(STATE_VEC_LEN, dtype=np.float32)
    state_mask = np.zeros(STATE_VEC_LEN, dtype=np.float32)

    for i, key in enumerate(arm_format.split(",")):
        state_vec[STATE_VEC_IDX_MAPPING[key]] = arm_concat[i]
        state_mask[STATE_VEC_IDX_MAPPING[key]] = 1

    return state_vec, state_mask


def state_vec_to_action(state_vec: np.ndarray) -> np.ndarray:
    batch_size = state_vec.shape[0]

    arm_format = "arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,arm_joint_6_pos,gripper_open,eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5"

    arm_concat = np.zeros((batch_size, 17), dtype=np.float32)
    for i, key in enumerate(arm_format.split(",")):
        arm_concat[:, i] = state_vec[:, STATE_VEC_IDX_MAPPING[key]]

    gripper_open = (arm_concat[:, 7:8]) * 2 - 1
    eef_pos = arm_concat[:, 8:11]
    eef_ang = arm_concat[:, 11:]

    eef_ang = ortho6d_to_rotation_matrix(eef_ang)
    eef_ang = R.from_matrix(eef_ang).as_euler("xyz")

    action = np.concatenate([eef_pos, eef_ang, gripper_open], axis=-1)
    return action


def parse_rgb_obs(rgb_obs: dict):
    rgb_static = Image.fromarray(rgb_obs["rgb_static"])
    rgb_gripper = Image.fromarray(rgb_obs["rgb_gripper"])
    return rgb_static, rgb_gripper, None


def rollout(env, model: RoboticDiffusionTransformerModel, task_oracle, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i):
    model: RoboticDiffusionTransformerModel

    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    # np.save(f"{eval_dir}/obs_{sequence_i}_{subtask_i}.npy", obs["robot_obs"])
    # exit(0)
    state_vec, state_mask = robot_obs_to_state_vec(obs["robot_obs"])
    lang_annotation = val_annotations[subtask][0]
    print("lang_annotation", lang_annotation)

    state_history = [state_vec, state_vec]
    state_mask_history = [state_mask, state_mask]
    rgb_obs_history = []
    rgb_obs_history.append(parse_rgb_obs(obs["rgb_obs"]))
    rgb_obs_history.append(parse_rgb_obs(obs["rgb_obs"]))

    replacements = {
        '_': ' ',
        '1f': ' ',
        '4f': ' ',
        '-': ' ',
        '50': ' ',
        '55': ' ',
        '56': ' ',
    }

    for key, value in replacements.items():
        lang_annotation = lang_annotation.replace(key, value)
    lang_annotation = lang_annotation.strip()
    lang_annotation = capitalize_and_period(lang_annotation)

    text_embeds = model.encode_instruction(lang_annotation)

    # model.reset()
    start_info = env.get_info()
    if debug:
        img_list = []
    horizon = 8
    for step in range(EP_LEN):
        if step % horizon == 0:
            state_vec = state_history[-1][None]
            state_mask = state_mask_history[-1][None]
            images = [
                rgb_obs_history[-2][0],
                rgb_obs_history[-2][1],
                rgb_obs_history[-2][2],
                rgb_obs_history[-1][0],
                rgb_obs_history[-1][1],
                rgb_obs_history[-1][2],
            ]
            # print(images)

            with torch.inference_mode():
                future_states = model.step(
                    state_vec=state_vec,
                    state_mask=state_mask,
                    images=images,
                    text_embeds=text_embeds,
                ).squeeze(0).cpu().numpy()

            actions = state_vec_to_action(future_states)

        obs, _, _, current_info = env.step(torch.from_numpy(actions[step % horizon]))
        state_vec, state_mask = robot_obs_to_state_vec(obs["robot_obs"])
        state_history.append(state_vec)
        state_mask_history.append(state_mask)
        rgb_obs_history.append(parse_rgb_obs(obs["rgb_obs"]))

        if debug:
            img_copy = copy.deepcopy(obs['rgb_obs']['rgb_static'])
            img_list.append(img_copy)
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
                clip = ImageSequenceClip(img_list, fps=30)
                clip.write_gif(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-succ.gif'), fps=30)
            return True
    if debug:
        print(colored("fail", "red"), end=" ")
        clip = ImageSequenceClip(img_list, fps=30)
        clip.write_gif(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-fail.gif'), fps=30)
    return False


# Initialize the model
import yaml

from calvin_model import create_model

def make_policy():
    with open("configs/base.yaml", "r") as fp:
        config = yaml.safe_load(fp)

    # pretrained_model_name_or_path = "robotics-diffusion-transformer/rdt-1b"
    # pretrained_model_name_or_path = "/mnt/dongxu-fs2/data-hdd/mingyang/projs/RoboticsDiffusionTransformer/checkpoints/rdt-finetune-calvin-1b/checkpoint-84000"
    # pretrained_model_name_or_path = "/mnt/dongxu-fs2/data-hdd/mingyang/projs/RoboticsDiffusionTransformer/checkpoints/rdt-finetune-calvin-170m"
    # pretrained_model_name_or_path = "/mnt/dongxu-fs2/data-hdd/mingyang/projs/RoboticsDiffusionTransformer/checkpoints/rdt-finetune-calvin-170m-v2/checkpoint-72000"
    pretrained_model_name_or_path = "/mnt/dongxu-fs2/data-hdd/mingyang/projs/RoboticsDiffusionTransformer/checkpoints/rdt-finetune-calvin-1b-v3/checkpoint-36000"
    # pretrained_model_name_or_path = "robotics-diffusion-transformer/rdt-1b"

    pretrained_text_encoder_name_or_path = "google/t5-v1_1-xxl"
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    model = create_model(
        args=config,
        dtype=torch.bfloat16,
        pretrained=pretrained_model_name_or_path,
        pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=30,
    )

    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_dir", default='task_ABCD_D/', type=str, help="Dataset directory.")
    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")
    parser.add_argument('--eval_dir', default="./eval_logs_rdt", type=str, help="Directory to save evaluation results")
    parser.add_argument('--device', default=0, type=int, help="CUDA device")
    args = parser.parse_args()

    seed_everything(0, workers=True)  # type:ignore

    device = torch.device('cuda', args.device)

    model = make_policy()
    # model = None

    observation_space = {
        'rgb_obs': ['rgb_static', 'rgb_gripper'],
        'depth_obs': [],
        'state_obs': ['robot_obs'],
        'actions': ['actions'],
        'language': ['language']}
    env = make_env(args.dataset_dir, observation_space, args.device)

    if not os.path.exists(args.eval_dir):
        os.makedirs(args.eval_dir, exist_ok=True)
    eval_sr_path = os.path.join(args.eval_dir, "success_rate.txt")
    eval_result_path = os.path.join(args.eval_dir, "result.txt")
    evaluate_policy(
        model,
        env,
        eval_sr_path,
        eval_result_path,
        args.eval_dir,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
