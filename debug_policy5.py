import numpy as np
import torch
from PIL import Image


def main():
    # obj = np.load("/home2/mingyang/projs/RoboticsDiffusionTransformer/cache2/chunk_0/sample_0.npz")
    # print(list(obj.keys()))

    # print(obj["state_chunk"].shape)
    # print(obj["action_chunk"].shape)

    # # print(obj["state_chunk"][-2:])
    # print(obj["past_frames_0"].shape, obj["past_frames_0"].dtype)

    # Image.fromarray(obj["past_frames_0"][0]).save("debug/rgb_past_frames_0_0.png")
    # Image.fromarray(obj["past_frames_0"][1]).save("debug/rgb_past_frames_0_1.png")

    # Image.fromarray(obj["past_frames_1"][0]).save("debug/rgb_past_frames_1_0.png")
    # Image.fromarray(obj["past_frames_1"][1]).save("debug/rgb_past_frames_1_1.png")
    # input_ids = torch.load("../RoboticsDiffusionTransformer/debug/sample_input_0.pt", map_location="cpu")
    # text_embeds = torch.load("../RoboticsDiffusionTransformer/debug/sample_text_embeds_0.pt", map_location="cpu")

    # print(input_ids.keys())
    # print(text_embeds.shape)


if __name__ == "__main__":
    main()
