import os
from pathlib import Path

import av
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image
from pytorch_lightning import seed_everything

from evaluate_calvin_rdt import make_env, capitalize_and_period
from evaluate_calvin_rdt import make_policy, robot_obs_to_state_vec, state_vec_to_action, parse_rgb_obs
from calvin_model import RoboticDiffusionTransformerModel
from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
)

CALVIN_ROOT = "/home/mingyang/projs/GR-1/calvin"


def inference_policy(
    policy: RoboticDiffusionTransformerModel,
    robot_obs,
    rgb_static,
    rgb_gripper,
    instruction,
):
    text_embeds = policy.encode_instruction(instruction)

    state_vec, state_mask = robot_obs_to_state_vec(robot_obs[0])
    images = parse_rgb_obs({
        "rgb_static": rgb_static[0],
        "rgb_gripper": rgb_gripper[0],
    }) * 2

    with torch.inference_mode():
        future_states = policy.step(
            state_vec=state_vec[None],
            state_mask=state_mask[None],
            images=images,
            text_embeds=text_embeds,
        ).squeeze(0).cpu().numpy()

    pred_actions = state_vec_to_action(future_states)

    return pred_actions


def save_to_video(frames, filename):
    # container = av.open(f"debug/pred_actions_{subtask_i}.mp4", mode="w")
    container = av.open(filename, mode="w")
    stream = container.add_stream("libx264", rate=30)
    stream.width = 200
    stream.height = 200
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23"}

    for frame in frames:
        frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "0"

    seed_everything(0, workers=True)  # type:ignore

    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_seqs = get_sequences(num_sequences=100, num_workers=1)

    observation_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
        "state_obs": ["robot_obs"],
        "actions": ["actions"],
        "language": ["language"]
    }
    root_path = Path("/mnt/dongxu-fs1/data-ssd/mingyang/datasets/CALVIN/task_ABCD_D/")
    env = make_env(
        str(root_path),
        observation_space,
        0,
    )

    policy: RoboticDiffusionTransformerModel = make_policy()

    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)

    scores = 0
    for eval_i, (initial_state, subtasks) in enumerate(eval_seqs):
        print("-" * 80)

        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)

        env.reset(scene_obs=scene_obs, robot_obs=robot_obs)

        all_frames = []
        for subtask_i, subtask in enumerate(subtasks):
            instr: str = val_annotations[subtask][0]

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
                instr = instr.replace(key, value)
            instr = instr.strip()
            instr = capitalize_and_period(instr)

            print("subtask", subtask, "|", instr)

            frames = []
            start_info = env.get_info()

            obs = env.get_obs()

            # Image.fromarray(obs["rgb_obs"]["rgb_static"]).save(f"debug/eval/rgb_static_{eval_i:03d}_{subtask_i}.png")

            done = False
            for chunk_i in range(10):
                if True:
                    obs = env.get_obs()

                    obs["instr"] = instr
                    obs["subtask"] = subtask
                    torch.save(obs, f"debug/eval_obs/obs_{eval_i:03d}_{subtask_i}_{chunk_i}.pt")

                    pred_actions = inference_policy(
                        policy,
                        [obs["robot_obs"]],
                        [obs["rgb_obs"]["rgb_static"]],
                        [obs["rgb_obs"]["rgb_gripper"]],
                        instr,
                    )

                    # np.save("eval_logs_rdt/pred_actions.npy", pred_actions)
                else:
                    pred_actions = np.load("eval_logs_rdt/pred_actions.npy")

                for step in tqdm(range(len(pred_actions))):
                # for step in tqdm(range(32)):
                    action = pred_actions[step]
                    obs, r, d, i = env.step(torch.from_numpy(action))

                    frames.append(obs["rgb_obs"]["rgb_static"])

                    current_task_info = task_oracle.get_task_info_for_set(start_info, i, {subtask})

                    if len(current_task_info) > 0:
                        done = True
                        print(current_task_info, "success")
                        break

                if done:
                    break

            all_frames.extend(frames)
            save_to_video(frames, f"debug/eval/pred_actions_{eval_i:03d}_{subtask_i}.mp4")

            if done:
                scores += 1
            else:
                break

        save_to_video(all_frames, f"debug/eval/pred_actions_{eval_i:03d}_all.mp4")

        print("+" * 80)
        print("Average success rate:", scores / (eval_i + 1))
        print("+" * 80)


if __name__ == "__main__":
    main()
