import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from configs.state_vec import STATE_VEC_IDX_MAPPING
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.rdt_runner import RDTRunner
from models.hdfree_runner import HDFreeRunner


# The indices that the raw vector should be mapped to in the unified action vector
AGILEX_STATE_INDICES = [
    STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING["left_gripper_open"]
] + [
    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING[f"right_gripper_open"]
]


# Create the RDT model
def create_model(args, **kwargs):
    model = RoboticDiffusionTransformerModel(args, **kwargs)
    pretrained = kwargs.get("pretrained", None)
    if (
        pretrained is not None
        and os.path.isfile(pretrained)
    ):
        model.load_pretrained_weights(pretrained)
    return model


class RoboticDiffusionTransformerModel(object):
    """A wrapper for the RDT model, which handles
            1. Model initialization
            2. Encodings of instructions
            3. Model inference
    """
    def __init__(
        self,
        args,
        device='cuda',
        dtype=torch.bfloat16,
        image_size=None,
        control_frequency=30,
        pretrained=None,
        pretrained_text_encoder_name_or_path=None,
        pretrained_vision_encoder_name_or_path=None,
        is_hdfree=False,
    ):
        self.args = args
        self.dtype = dtype
        self.image_size = image_size
        self.device = device
        self.control_frequency = control_frequency
        self.text_tokenizer, self.text_model = self.get_text_encoder(pretrained_text_encoder_name_or_path)
        self.image_processor, self.vision_model = self.get_vision_encoder(pretrained_vision_encoder_name_or_path)
        self.is_hdfree = is_hdfree

        self.policy = self.get_policy(pretrained)

        self.reset()
        print(f"[DEBUG] model loaded. is_hdfree: {self.is_hdfree}")

    def get_policy(self, pretrained):
        """Initialize the model."""
        # Initialize model with arguments
        if not self.is_hdfree:
            # Original RDT
            if (
                pretrained is None
                or os.path.isfile(pretrained)
            ):
                img_cond_len = (
                    self.args["common"]["img_history_size"]
                    * self.args["common"]["num_cameras"]
                    * self.vision_model.num_patches
                )

                _model = RDTRunner(
                    action_dim=self.args["common"]["state_dim"],
                    pred_horizon=self.args["common"]["action_chunk_size"],
                    config=self.args["model"],
                    lang_token_dim=self.args["model"]["lang_token_dim"],
                    img_token_dim=self.args["model"]["img_token_dim"],
                    state_token_dim=self.args["model"]["state_token_dim"],
                    max_lang_cond_len=self.args["dataset"]["tokenizer_max_length"],
                    img_cond_len=img_cond_len,
                    img_pos_embed_config=[
                        # No initial pos embed in the last grid size
                        # since we've already done in ViT
                        ("image", (self.args["common"]["img_history_size"],
                            self.args["common"]["num_cameras"],
                            -self.vision_model.num_patches)),
                    ],
                    lang_pos_embed_config=[
                        # Similarly, no initial pos embed for language
                        ("lang", -self.args["dataset"]["tokenizer_max_length"]),
                    ],
                    dtype=self.dtype,
                )
                print("[RoboticDiffusionTransformerModel] RDTRunner loaded from scratch.")
            else:
                _model = RDTRunner.from_pretrained(pretrained)
                print(f"[RoboticDiffusionTransformerModel] RDTRunner loaded from: {pretrained}.")

        else:
            # HDFree version of RDT, registered with ia3_adapters
            img_cond_len = (
                    self.args["common"]["img_history_size"]
                    * self.args["common"]["num_cameras"]
                    * self.vision_model.num_patches
            )

            _model = HDFreeRunner(
                action_dim=self.args["common"]["state_dim"],
                pred_horizon=self.args["common"]["action_chunk_size"],
                config=self.args["model"],
                lang_token_dim=self.args["model"]["lang_token_dim"],
                img_token_dim=self.args["model"]["img_token_dim"],
                state_token_dim=self.args["model"]["state_token_dim"],
                max_lang_cond_len=self.args["dataset"]["tokenizer_max_length"],
                img_cond_len=img_cond_len,
                img_pos_embed_config=[
                    # No initial pos embed in the last grid size
                    # since we've already done in ViT
                    ("image", (self.args["common"]["img_history_size"],
                        self.args["common"]["num_cameras"],
                        -self.vision_model.num_patches)),
                ],
                lang_pos_embed_config=[
                    # Similarly, no initial pos embed for language
                    ("lang", -self.args["dataset"]["tokenizer_max_length"]),
                ],
                dtype=self.dtype,
            )
            _model.register_modules_for_target()

            from safetensors.torch import load_model
            ema_weight = os.path.join(pretrained, "ema", "model.safetensors")
            load_model(_model, ema_weight)

            for name, p in _model.named_parameters():
                if "ia3" in name:
                    print(name)
            print(f"[RoboticDiffusionTransformerModel] HDFreeRunner loaded from: {ema_weight}.")

        return _model

    def get_text_encoder(self, pretrained_text_encoder_name_or_path):
        text_embedder = T5Embedder(
            from_pretrained=pretrained_text_encoder_name_or_path,
            model_max_length=self.args["dataset"]["tokenizer_max_length"],
            device=self.device,
        )
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
        return tokenizer, text_encoder

    def get_vision_encoder(self, pretrained_vision_encoder_name_or_path):
        vision_encoder = SiglipVisionTower(vision_tower=pretrained_vision_encoder_name_or_path, args=None)
        image_processor = vision_encoder.image_processor
        return image_processor, vision_encoder

    def reset(self):
        """Set model to evaluation mode.
        """
        device = self.device
        weight_dtype = self.dtype
        self.policy.eval()
        self.text_model.eval()
        self.vision_model.eval()

        self.policy = self.policy.to(device, dtype=weight_dtype)
        self.text_model = self.text_model.to(device, dtype=weight_dtype)
        self.vision_model = self.vision_model.to(device, dtype=weight_dtype)

    def load_pretrained_weights(self, pretrained=None):
        if pretrained is None:
            return
        print(f'Loading weights from {pretrained}')
        filename = os.path.basename(pretrained)
        if filename.endswith('.pt'):
            checkpoint =  torch.load(pretrained)
            self.policy.load_state_dict(checkpoint["module"])
        elif filename.endswith('.safetensors'):
            from safetensors.torch import load_model
            load_model(self.policy, pretrained)
        else:
            raise NotImplementedError(f"Unknown checkpoint format: {pretrained}")

    def encode_instruction(self, instruction, device=None):
        """Encode string instruction to latent embeddings.

        Args:
            instruction: a string of instruction
            device: a string of device

        Returns:
            pred: a tensor of latent embeddings of shape (text_max_length, 512)
        """
        tokens = self.text_tokenizer(
            instruction, return_tensors="pt",
            padding="longest",
            truncation=True
        )["input_ids"].to(self.device)

        tokens = tokens.view(1, -1)
        with torch.no_grad():
            pred = self.text_model(tokens).last_hidden_state.detach()

        return pred

    @torch.no_grad()
    def step(self, state_vec, state_mask, images, text_embeds):
        """
        Predict the next action chunk given the
        proprioceptive states, images, and instruction embeddings.

        Args:
            proprio: proprioceptive states
            images: RGB images, the order should be
                [ext_{t-1}, right_wrist_{t-1}, left_wrist_{t-1},
                ext_{t}, right_wrist_{t}, left_wrist_{t}]
            text_embeds: instruction embeddings

        Returns:
            action: predicted action
        """
        device = self.device
        dtype = self.dtype

        # The background image used for padding
        background_color = np.array([
            int(x * 255) for x in self.image_processor.image_mean
        ], dtype=np.uint8).reshape(1, 1, 3)
        background_image = np.ones((
            self.image_processor.size["height"],
            self.image_processor.size["width"],
            3,
        ), dtype=np.uint8) * background_color

        # Preprocess the images by order and encode them
        image_tensor_list = []
        for image in images:
            if image is None:
                # Replace it with the background image
                image = Image.fromarray(background_image)

            if self.image_size is not None:
                image = transforms.Resize(self.data_args.image_size)(image)

            if self.args["dataset"].get("auto_adjust_image_brightness", False):
                pixel_values = list(image.getdata())
                average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                if average_brightness <= 0.15:
                    image = transforms.ColorJitter(brightness=(1.75,1.75))(image)

            if self.args["dataset"].get("image_aspect_ratio", "pad") == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            image_tensor_list.append(image)

        image_tensor = torch.stack(image_tensor_list, dim=0).to(device, dtype=dtype)
        # ????????????????????????????????????????????????????????
        # image_tensor = torch.flip(image_tensor, [1])  # BGR to RGB

        # torch.save(image_tensor, "eval_logs_rdt/image_tensor.pt")

        # tmp = image_tensor
        # batch = torch.load("../RoboticsDiffusionTransformer/debug/sample_batch_0.pt", map_location="cpu")
        # image_tensor = batch["images"].to(device, dtype=dtype)
        # image_tensor = image_tensor.flatten(0, 1)

        # print("image_tensor", image_tensor.shape)
        # diff = torch.abs(tmp - image_tensor)
        # print("image_tensor diff")
        # print("diff", torch.mean(diff))
        # print("diff", torch.std(diff))
        # print("diff", torch.max(diff))
        # print("diff", torch.min(diff))
        # print("+" * 80)

        image_embeds = self.vision_model(image_tensor).detach()
        # torch.save(image_embeds, "eval_logs_rdt/image_embeds.pt")
        image_embeds = image_embeds.reshape(-1, self.vision_model.hidden_size).unsqueeze(0)

        # Prepare the proprioception states and the control frequency
        # joints = proprio.to(device).unsqueeze(0)   # (1, 1, 14)
        # states, state_elem_mask = self._format_joint_to_state(joints)    # (1, 1, 128), (1, 128)
        # states, state_elem_mask = states.to(device, dtype=dtype), state_elem_mask.to(device, dtype=dtype)
        # states = states[:, -1:, :]  # (1, 1, 128)
        states = torch.from_numpy(state_vec).to(device, dtype=dtype).unsqueeze(1)
        state_mask = torch.from_numpy(state_mask).to(device, dtype=dtype).unsqueeze(1)
        ctrl_freqs = torch.tensor([self.control_frequency]).to(device)

        text_embeds = text_embeds.to(device, dtype=dtype)

        lang_attn_mask = torch.ones(
            text_embeds.shape[:2], dtype=torch.bool,
            device=text_embeds.device,
        )

        # obj = torch.load("../RoboticsDiffusionTransformer/debug/sample_input_0.pt", map_location="cpu")
        # text_embeds = obj["lang_tokens"].to(device, dtype=dtype)
        # lang_attn_mask = obj["lang_attn_mask"].to(device)
        # tmp = image_embeds
        # image_embeds = obj["img_tokens"].to(device, dtype=dtype)
        # tmp2 = states
        # states = obj["state_tokens"].to(device, dtype=dtype)
        # state_mask = obj["action_mask"].to(device, dtype=dtype)
        # ctrl_freqs = obj["ctrl_freqs"].to(device)

        # diff = torch.abs(tmp - image_embeds)
        # print("image_embeds diff", tmp.shape, image_embeds.shape)
        # print("diff", torch.mean(diff))
        # print("diff", torch.std(diff))
        # print("diff", torch.max(diff))
        # print("diff", torch.min(diff))
        # print("+" * 80)
        # image_embeds = tmp

        # diff = torch.abs(tmp2 - states)
        # print("states diff")
        # print("diff", torch.mean(diff))
        # print("diff", torch.std(diff))
        # print("diff", torch.max(diff))
        # print("diff", torch.min(diff))
        # print("+" * 80)

        # self.policy.eval()

        # Predict the next action chunk given the inputs
        trajectory = self.policy.predict_action(
            lang_tokens=text_embeds,
            lang_attn_mask=lang_attn_mask,
            img_tokens=image_embeds,
            state_tokens=states,
            action_mask=state_mask,
            ctrl_freqs=ctrl_freqs,
        )
        # trajectory = self._unformat_action_to_joint(trajectory).to(torch.float32)
        trajectory = trajectory.to(torch.float32)

        # pred_actions = torch.load("../RoboticsDiffusionTransformer/debug/sample_pred_actions_0.pt", map_location="cpu")
        # pred_actions = pred_actions.to(device, dtype=dtype)
        # print("pred_actions", pred_actions.shape)

        # diff = torch.abs(trajectory - pred_actions)
        # print("trajectory diff")
        # print("diff", torch.mean(diff))
        # print("diff", torch.std(diff))
        # print("diff", torch.max(diff))
        # print("diff", torch.min(diff))
        # print("+" * 80)

        return trajectory
