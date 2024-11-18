import torch

from models.multimodal_encoder.siglip_encoder import SiglipVisionTower


def main():
    vision_encoder = SiglipVisionTower(vision_tower="google/siglip-so400m-patch14-384", args=None)
    vision_encoder.eval()

    # vision_encoder.load_state_dict(torch.load("../RoboticsDiffusionTransformer/debug/vision_encoder.pt", map_location="cpu"))

    vision_encoder.to("cuda", dtype=torch.bfloat16)
    device = "cuda"
    dtype = torch.bfloat16

    batch = torch.load("../RoboticsDiffusionTransformer/debug/sample_images_0.pt", map_location="cpu")
    image_tensor = batch.to(device, dtype=dtype)
    # image_tensor = batch["images"].to(device, dtype=dtype)
    # image_tensor = image_tensor.flatten(0, 1)

    print("image_tensor", image_tensor.shape)

    with torch.no_grad():
        image_embeds = vision_encoder(image_tensor)
        image_embeds = image_embeds.reshape(-1, vision_encoder.hidden_size).unsqueeze(0)

    print("image_embeds", image_embeds.shape)

    # obj = torch.load("../RoboticsDiffusionTransformer/debug/sample_input_0.pt", map_location="cpu")
    # tmp = image_embeds
    # image_embeds = obj["img_tokens"].to(device, dtype=dtype)
    tmp = image_embeds
    image_embeds = torch.load("../RoboticsDiffusionTransformer/debug/sample_image_embeds_0.pt", map_location="cpu")
    image_embeds = image_embeds.to(device, dtype=dtype)

    diff = torch.abs(tmp - image_embeds)
    print("diff", diff.mean().item())
    print("diff", diff.std().item())
    print("diff", diff.max().item())
    print("diff", diff.min().item())

    print("*" * 80)

    with torch.no_grad():
        image_embeds = vision_encoder(image_tensor)
        image_embeds = image_embeds.reshape(-1, vision_encoder.hidden_size).unsqueeze(0)

    diff = torch.abs(tmp - image_embeds)
    print("diff", diff.mean().item())
    print("diff", diff.std().item())
    print("diff", diff.max().item())
    print("diff", diff.min().item())


if __name__ == "__main__":
    main()
