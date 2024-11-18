from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm

from evaluate_calvin_rdt import make_policy, robot_obs_to_state_vec, state_vec_to_action, parse_rgb_obs
from calvin_model import RoboticDiffusionTransformerModel

CALVIN_ROOT = "/home/mingyang/projs/GR-1/calvin"


def main():
    batch = torch.load("../RoboticsDiffusionTransformer/debug/sample_batch_0.pt", map_location="cpu")
    pred_actions = torch.load("../RoboticsDiffusionTransformer/debug/sample_pred_actions_0.pt", map_location="cpu")
    text_embeds = torch.load("../RoboticsDiffusionTransformer/debug/sample_text_embeds_0.pt", map_location="cpu")

    print(pred_actions.shape)
    print(text_embeds.shape)

    print(batch.keys())

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)
        else:
            print(k, len(v))

    print("data_indices", batch["data_indices"])

    state_elem_mask = batch["state_elem_mask"]
    num_steps = 64
    expanded_state_elem_mask = state_elem_mask.unsqueeze(1).tile((1, num_steps, 1)).float()

    loss = F.mse_loss(pred_actions, batch["actions"], reduction="none")
    mse_loss_per_entry = ((loss * expanded_state_elem_mask).reshape((1, -1)).sum(1)
                            / expanded_state_elem_mask.reshape((1, -1)).sum(1))
    print("mse", mse_loss_per_entry.item())

    # print(pred_actions[:10])
    # print(batch["actions"][:10])

    pred_actions = state_vec_to_action(pred_actions[0].float().numpy())
    gt_actions = state_vec_to_action(batch["actions"][0].float().numpy())

    loss = F.mse_loss(torch.from_numpy(pred_actions), torch.from_numpy(gt_actions))
    print("mse", loss.item())

    states = state_vec_to_action(batch["states"][0].float().numpy())
    print("states", states[:4])

    print("pred_actions", pred_actions[:4])
    print("gt_actions", gt_actions[:4])
    print("gt_actions", gt_actions[-4:])

    # return

    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)

    root_path = Path("/mnt/dongxu-fs1/data-ssd/mingyang/datasets/CALVIN/task_ABCD_D/")

    annos = np.load(root_path / "training" / "lang_annotations" / "auto_lang_ann.npy", allow_pickle=True)
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
        obj = np.load(root_path / "training" / f"episode_{i:07d}.npz")
        actions.append(obj["actions"])
        rel_actions.append(obj["rel_actions"])
        robot_obs.append(obj["robot_obs"])
        rgb_static.append(obj["rgb_static"])
        rgb_gripper.append(obj["rgb_gripper"])
        scene_obs.append(obj["scene_obs"])

    policy: RoboticDiffusionTransformerModel = make_policy()

    text_embeds = policy.encode_instruction(instruction)

    torch.save(text_embeds, "eval_logs_rdt/text_embeds.pt")
    # text_embeds = torch.load("eval_logs_rdt/text_embeds.pt")

    print("text_embeds", text_embeds.shape)

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

    np.save("eval_logs_rdt/pred_actions.npy", pred_actions)

    gt_actions = np.stack(actions[:64], axis=0)

    diff = np.abs(pred_actions - gt_actions)
    # print(diff)
    print(np.mean(diff))
    print(np.std(diff))
    print(np.max(diff))
    print(np.min(diff))

    print("pred_actions", pred_actions[:10])
    print("gt_actions", gt_actions[:10])

    mse = F.mse_loss(torch.from_numpy(pred_actions), torch.from_numpy(gt_actions))
    print("mse", mse.item())


if __name__ == "__main__":
    main()
