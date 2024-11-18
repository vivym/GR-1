from pathlib import Path

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm

from evaluate_calvin_rdt import make_env

CALVIN_ROOT = "/home/mingyang/projs/GR-1/calvin"


def main():
    obj = np.load("../RoboticsDiffusionTransformer/debug/data_rotate.npz")

    actions = obj["actions"]
    scene_obs = obj["scene_obs"]
    robot_obs = obj["robot_obs"]
    subtask = obj["subtask"].item()

    print(subtask, type(subtask))

    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)

    root_path = Path("/mnt/dongxu-fs1/data-ssd/mingyang/datasets/CALVIN/task_ABCD_D/")

    observation_space = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "depth_obs": [],
        "state_obs": ["robot_obs"],
        "actions": ["actions"],
        "language": ["language"]
    }
    env = make_env(
        str(root_path),
        observation_space,
        0,
    )

    env.reset(scene_obs=scene_obs[0], robot_obs=robot_obs[0])

    start_info = env.get_info()

    for step in tqdm(range(len(actions))):
        action = actions[step]
        obs, r, d, i = env.step(torch.from_numpy(action))

        # print(obs.keys())
        # diff = np.abs(obs["robot_obs"] - robot_obs[step])
        # print(diff)

        current_task_info = task_oracle.get_task_info_for_set(start_info, i, {subtask})

        if len(current_task_info) > 0:
            print(current_task_info, "success")
            break



if __name__ == "__main__":
    main()
