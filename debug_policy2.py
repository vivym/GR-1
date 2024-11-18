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
    obj = np.load("../RoboticsDiffusionTransformer/debug/calvin_data.npz")
    gt_actions = obj["action"]
    robot_obs = obj["robot_obs"]
    rgb_static = obj["rgb_static"]
    rgb_gripper = obj["rgb_gripper"]
    instruction = obj["instruction"][0].item().decode("utf-8")

    print(gt_actions.shape)
    print(robot_obs.shape)
    print(rgb_static.shape)
    print(rgb_gripper.shape)
    print(instruction, type(instruction))

    state_vec, state_mask = robot_obs_to_state_vec(robot_obs[0])

    # print(gt_actions[:4])
    # print(gt_actions[-4:])
    # return

    policy: RoboticDiffusionTransformerModel = make_policy()

    text_embeds = policy.encode_instruction(instruction)

    torch.save(text_embeds, "eval_logs_rdt/text_embeds.pt")
    # text_embeds = torch.load("eval_logs_rdt/text_embeds.pt")

    print("text_embeds", text_embeds.shape)

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

    gt_actions = np.stack(gt_actions[1:], axis=0)

    diff = np.abs(pred_actions[..., :-1] - gt_actions[..., :-1])
    # print(diff)
    print(np.mean(diff))
    print(np.std(diff))
    print(np.max(diff))
    print(np.min(diff))

    print("pred_actions", pred_actions[:4])
    print("gt_actions", gt_actions[:4])

    mse = F.mse_loss(torch.from_numpy(pred_actions)[..., :-1], torch.from_numpy(gt_actions)[..., :-1])
    print("mse", mse.item())


if __name__ == "__main__":
    main()
