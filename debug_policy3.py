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
    instruction = "Move the door all the way to the right."

    print(gt_actions.shape)
    print(robot_obs.shape)
    print(rgb_static.shape)
    print(rgb_gripper.shape)
    print(instruction, type(instruction))

    state_vec, state_mask = robot_obs_to_state_vec(robot_obs[0])

    batch = torch.load("../RoboticsDiffusionTransformer/debug/sample_batch_0.pt", map_location="cpu")
    pred_actions = torch.load("../RoboticsDiffusionTransformer/debug/sample_pred_actions_0.pt", map_location="cpu")
    text_embeds = torch.load("../RoboticsDiffusionTransformer/debug/sample_text_embeds_0.pt", map_location="cpu")

    print(pred_actions.shape)
    print(text_embeds.shape)

    print(batch.keys())

    print(gt_actions[:4])
    print(robot_obs[:4])

    actions_in_batch = state_vec_to_action(batch["actions"][0].float().numpy())
    states_in_batch = state_vec_to_action(batch["states"][0].float().numpy())

    print("states", states_in_batch[-1])
    print("actions", actions_in_batch[:4])

    offset = 3

    batch["states"][0][0]
    state_vec, state_mask = robot_obs_to_state_vec(robot_obs[0 + offset])

    diff = np.abs(batch["states"][0][-1].float().numpy() - state_vec)
    # print("diff", diff)
    print("diff", np.mean(diff))
    print("diff", np.std(diff))
    print("diff", np.max(diff))
    print("diff", np.min(diff))

    tmp = state_vec_to_action(batch["states"][0][-1:])
    diff = np.abs(tmp[0] - gt_actions[0 + offset - 1])
    print("tmp", tmp[0])
    print("gt_actions", gt_actions[0 + offset - 1])
    print("diff", diff)
    print("diff", np.mean(diff))
    print("diff", np.std(diff))
    print("diff", np.max(diff))
    print("diff", np.min(diff))

    images = parse_rgb_obs({
        "rgb_static": rgb_static[0 + offset],
        "rgb_gripper": rgb_gripper[0 + offset],
    }) * 2

    print(batch["images"].shape)

    policy: RoboticDiffusionTransformerModel = make_policy()

    text_embeds2 = policy.encode_instruction(instruction)

    torch.save(text_embeds2, "eval_logs_rdt/text_embeds.pt")
    # text_embeds = torch.load("eval_logs_rdt/text_embeds.pt")

    print("text_embeds", text_embeds.shape)
    print("text_embeds2", text_embeds2.shape)

    print("state_elem_mask", batch["state_elem_mask"].shape)

    diff = torch.abs(text_embeds - text_embeds2.to(text_embeds.device, dtype=text_embeds.dtype))
    print("text_embeds diff", torch.mean(diff))
    print("text_embeds diff", torch.std(diff))
    print("text_embeds diff", torch.max(diff))
    print("text_embeds diff", torch.min(diff))

    # print("text_embeds", text_embeds.shape)

    with torch.inference_mode():
        future_states = policy.step(
            # state_vec=state_vec[None],
            # state_mask=state_mask[None],
            state_vec=batch["states"][0][-1][None].numpy(),
            state_mask=batch["state_elem_mask"][0][None].numpy(),
            images=images,
            text_embeds=text_embeds,
        ).squeeze(0).cpu().numpy()

    gt_pred_actions = pred_actions

    diff = np.abs(gt_pred_actions.float().squeeze(0).cpu().numpy() - future_states)
    # print(diff)
    print("pred_actions diff")
    print(np.mean(diff))
    print(np.std(diff))
    print(np.max(diff))
    print(np.min(diff))
    print("-" * 80)

    pred_actions = state_vec_to_action(future_states)

    np.save("eval_logs_rdt/pred_actions.npy", pred_actions)

    # gt_actions = np.stack(gt_actions[1:], axis=0)
    gt_actions = actions_in_batch

    diff = np.abs(pred_actions[..., :-1] - gt_actions[..., :-1])
    # print(diff)
    print(np.mean(diff))
    print(np.std(diff))
    print(np.max(diff))
    print(np.min(diff))

    print("pred_actions", pred_actions[:4])
    print("gt_actions", gt_actions[:4])

    print("pred_actions", pred_actions[-4:])
    print("gt_actions", gt_actions[-4:])

    mse = F.mse_loss(torch.from_numpy(pred_actions)[..., :-1], torch.from_numpy(gt_actions)[..., :-1])
    print("mse", mse.item())


if __name__ == "__main__":
    main()
