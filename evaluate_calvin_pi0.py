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
import inspect
from typing import Optional
from moviepy.editor import ImageSequenceClip
from scipy.spatial.transform import Rotation as R
from PIL import Image

# This is for using the locally installed repo clone when using slurm
# from calvin_agent.models.calvin_base_model import CalvinBaseModel
# from calvin_model import RoboticDiffusionTransformerModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from models.pi0_gemma import (
    Pi0GemmaForConditionalGeneration, Pi0GemmaForConditionalGenerationOutputWithPast, Pi0GemmaProcessor
)

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


def retrieve_timesteps(
    scheduler: FlowMatchEulerDiscreteScheduler,
    num_inference_steps: int | None = None,
    device: Optional[str | torch.device] = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def make_env(dataset_path, observation_space, device_id):
    val_folder = Path(dataset_path) / "validation"
    from evaluation.calvin_env_wrapper_raw import CalvinEnvWrapperRaw
    device = torch.device('cuda', device_id)
    env = CalvinEnvWrapperRaw(val_folder, observation_space, device)
    return env


def evaluate_policy(model, preprocessor, noise_scheduler, env, eval_sr_path, eval_result_path, eval_dir=None, debug=False):
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
        result = evaluate_sequence(env, model, preprocessor, noise_scheduler, task_oracle, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i)
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


def evaluate_sequence(env, model, preprocessor, noise_scheduler, task_checker, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i):
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
        success = rollout(env, model, preprocessor, noise_scheduler, task_checker, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i)
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


def parse_rgb_obs(rgb_obs: dict):
    rgb_static = Image.fromarray(rgb_obs["rgb_static"])
    rgb_gripper = Image.fromarray(rgb_obs["rgb_gripper"])
    return rgb_static, rgb_gripper, None


def rollout(env, model: Pi0GemmaForConditionalGeneration, preprocessor: Pi0GemmaProcessor, noise_scheduler, task_oracle, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i):
    device = torch.device("cuda")
    weight_dtype = torch.float32
    device = next(model.parameters()).device
    weight_dtype = next(model.parameters()).dtype

    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    # np.save(f"{eval_dir}/obs_{sequence_i}_{subtask_i}.npy", obs["robot_obs"])
    # exit(0)
    # state_vec, state_mask = robot_obs_to_state_vec(obs["robot_obs"])
    lang_annotation = val_annotations[subtask][0]
    print("lang_annotation", lang_annotation)

    # model.reset()
    start_info = env.get_info()
    if debug:
        img_list = []
    horizon = 32
    for step in range(EP_LEN):
        if step % horizon == 0:

            with torch.inference_mode():
                images = [obs["rgb_obs"]["rgb_static"], obs["rgb_obs"]["rgb_gripper"]]
                sample = preprocessor.prepare_for_traning_sample(
                    images=images,
                    instruction=lang_annotation,
                    propri_states=torch.from_numpy(obs["robot_obs"][None]).to(torch.float32),
                )

                batch = preprocessor.collate([sample])

                input_ids: torch.Tensor = batch["input_ids"]
                attention_mask: torch.Tensor = batch["attention_mask"]
                pixel_values: torch.Tensor = batch["pixel_values"]
                propri_states: torch.Tensor = batch["propri_states"]

                input_ids = input_ids.to(device=device)
                attention_mask = attention_mask.to(device=device)
                pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
                propri_states = propri_states.to(device=device, dtype=weight_dtype)

                num_inference_steps = 50

                timesteps, num_inference_steps = retrieve_timesteps(noise_scheduler, num_inference_steps, device, None)
                num_warmup_steps = max(len(timesteps) - num_inference_steps * noise_scheduler.order, 0)

                pred_actions = torch.randn(1, 64, 7, dtype=weight_dtype, device=device)

                with tqdm(range(num_inference_steps), desc="Inference steps", disable=True) as progress_bar:
                    for i, t in enumerate(timesteps):
                        t: torch.Tensor
                        timestep = t.expand(pred_actions.shape[0])

                        outputs: Pi0GemmaForConditionalGenerationOutputWithPast = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            propri_states=propri_states,
                            timesteps=timestep,
                            noisy_actions=pred_actions,
                            return_dict=True,
                        )
                        model_pred = outputs.model_pred

                        pred_actions = noise_scheduler.step(model_pred, t, pred_actions, return_dict=False)[0]

                        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % noise_scheduler.order == 0):
                            progress_bar.update()

                actions = pred_actions.cpu().numpy()[0]

        obs, _, _, current_info = env.step(torch.from_numpy(actions[step % horizon]))

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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_dir", default='task_ABCD_D/', type=str, help="Dataset directory.")
    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")
    parser.add_argument('--eval_dir', default="./eval_logs_pi0_v4", type=str, help="Directory to save evaluation results")
    parser.add_argument('--device', default=0, type=int, help="CUDA device")
    args = parser.parse_args()

    seed_everything(0, workers=True)  # type:ignore

    device = torch.device('cuda', args.device)

    # model = make_policy()
    # model = None
    preprocessor: Pi0GemmaProcessor = Pi0GemmaProcessor.from_pretrained(
        "../OpenPi0/weights/pi0gemma-3b-mix-224-initial",
    )

    noise_scheduler: FlowMatchEulerDiscreteScheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        "../OpenPi0/weights/pi0gemma-3b-mix-224-initial"
    )

    model: Pi0GemmaForConditionalGeneration = Pi0GemmaForConditionalGeneration.from_pretrained(
        "/mnt/dongxu-fs2/data-hdd/mingyang/projs/OpenPi0/outputs/pi0-gemma-3b/step_00010000/ckpt",
    )
    model.eval()
    model.to(device=device)
    # model = None

    observation_space = {
        'rgb_obs': ['rgb_static', 'rgb_gripper'],
        'depth_obs': [],
        'state_obs': ['robot_obs'],
        'actions': ['rel_actions'],
        'language': ['language']}
    env = make_env(args.dataset_dir, observation_space, args.device)

    if not os.path.exists(args.eval_dir):
        os.makedirs(args.eval_dir, exist_ok=True)
    eval_sr_path = os.path.join(args.eval_dir, "success_rate.txt")
    eval_result_path = os.path.join(args.eval_dir, "result.txt")
    evaluate_policy(
        model,
        preprocessor,
        noise_scheduler,
        env,
        eval_sr_path,
        eval_result_path,
        args.eval_dir,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
