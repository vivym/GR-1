import os
from pathlib import Path

import av
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm

from evaluate_calvin_rdt import make_env
from evaluate_calvin_rdt import make_policy, robot_obs_to_state_vec, state_vec_to_action, parse_rgb_obs
from calvin_model import RoboticDiffusionTransformerModel

CALVIN_ROOT = "/home/mingyang/projs/GR-1/calvin"


def inference_policy(
    robot_obs,
    rgb_static,
    rgb_gripper,
    instruction,
):
    policy: RoboticDiffusionTransformerModel = make_policy()

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


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "0"

    obj = np.load("../RoboticsDiffusionTransformer/debug/data_rotate.npz")

    actions = obj["actions"]
    scene_obs = obj["scene_obs"]
    robot_obs = obj["robot_obs"]
    rgb_static = obj["rgb_static"]
    rgb_gripper = obj["rgb_gripper"]
    instr = obj["instr"].item()
    subtask = obj["subtask"].item()

    print(subtask, type(subtask))
    print(instr, type(instr))

    subtask = "rotate_pink_block_right"
    # instr = "Grasp the blue block, then hold it, and rotate it to the right."
    instr = "Take the pink block and rotate it right."

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

    env.reset(scene_obs=scene_obs[0], robot_obs=robot_obs[0])

    obs = env.get_obs()

    if True:
        pred_actions = inference_policy(
            robot_obs,
            [obs["rgb_obs"]["rgb_static"]],
            [obs["rgb_obs"]["rgb_gripper"]],
            instr,
        )

        np.save("eval_logs_rdt/pred_actions.npy", pred_actions)
    else:
        pred_actions = np.load("eval_logs_rdt/pred_actions.npy")

    print("pred_actions", pred_actions.shape)

    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)

    start_info = env.get_info()

    frames = []
    for step in tqdm(range(len(pred_actions))):
        action = pred_actions[step]
        obs, r, d, i = env.step(torch.from_numpy(action))

        # print(obs.keys())
        frames.append(obs["rgb_obs"]["rgb_static"])

        # print(obs.keys())
        # diff = np.abs(obs["robot_obs"] - robot_obs[step])
        # print(diff)

        current_task_info = task_oracle.get_task_info_for_set(start_info, i, {subtask})

        if len(current_task_info) > 0:
            print(current_task_info, "success")
            break

    container = av.open("debug/pred_actions.mp4", mode="w")
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


if __name__ == "__main__":
    main()
