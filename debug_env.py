from pathlib import Path

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm

from evaluate_calvin_rdt import make_env

CALVIN_ROOT = "/home/mingyang/projs/GR-1/calvin"


def main():
    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)

    root_path = Path("/mnt/dongxu-fs1/data-ssd/mingyang/datasets/CALVIN/task_ABCD_D/")

    annos = np.load(root_path / "validation" / "lang_annotations" / "auto_lang_ann.npy", allow_pickle=True)
    annos = annos.item()
    subtask = annos["language"]["task"][0]
    print(annos["language"]["task"][0])
    # print(annos["language"]["ann"][0])
    # print(annos["info"]["indx"][0])

    start_idx, end_idx = annos["info"]["indx"][0]

    instruction = annos["language"]["ann"][0]

    print(instruction)

    print(start_idx, end_idx, end_idx - start_idx)

    actions = []
    rel_actions = []
    robot_obs = []
    rgb_static = []
    rgb_gripper = []
    scene_obs = []
    for i in range(start_idx, end_idx):
        obj = np.load(root_path / "validation" / f"episode_{i:07d}.npz")
        actions.append(obj["actions"])
        rel_actions.append(obj["rel_actions"])
        robot_obs.append(obj["robot_obs"])
        rgb_static.append(obj["rgb_static"])
        rgb_gripper.append(obj["rgb_gripper"])
        scene_obs.append(obj["scene_obs"])

    observation_space = {
        'rgb_obs': ['rgb_static', 'rgb_gripper'],
        'depth_obs': [],
        'state_obs': ['robot_obs'],
        'actions': ['actions'],
        'language': ['language']
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

        print(obs.keys())
        diff = np.abs(obs["robot_obs"] - robot_obs[step])
        print(diff)

        current_task_info = task_oracle.get_task_info_for_set(start_info, i, {subtask})

        if len(current_task_info) > 0:
            print(current_task_info)
            break


if __name__ == "__main__":
    main()
