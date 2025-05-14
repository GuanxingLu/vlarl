import torch
from torch import nn
import os
from peft import PeftModel
import numpy as np
from PIL import Image
from termcolor import cprint
from accelerate.utils import is_peft_model


## Note that the following code is modified from
## https://github.com/microsoft/DeepSpeedExamples/blob/fd79b31848d9a46ceb63fbefe1f9603da8a275b9/applications/DeepSpeed-Chat/dschat/utils/model/reward_model.py#L11
class CriticVLA(nn.Module):

    def __init__(self, cfg, base_model, adapter_dir=None, pad_token_id='32000', num_padding_at_beginning=0):
        super().__init__()
        self.cfg = cfg
        self.config = base_model.config.text_config
        # self.config = base_model.get_model_config().text_config
        self.num_padding_at_beginning = num_padding_at_beginning
        # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
        self.config.n_embd = self.config.hidden_size if hasattr(
            self.config, "hidden_size") else self.config.n_embd

        self.v_head_mlp1 = nn.Linear(self.config.n_embd, 1024, bias=False)
        self.v_head_mlp2 = nn.Linear(1024, 512, bias=False)
        self.v_head_mlp3 = nn.Linear(512, 1, bias=False)
        self.relu = nn.ReLU()

        self.rwtranrsformer = base_model

        # freeze the rwtranrsformer (caution: fsdp auto-wrap error)
        if not is_peft_model(self.rwtranrsformer):
            for param in self.rwtranrsformer.parameters():
                param.requires_grad = False

        # load critic.pth in adapter_dir as linear layer
        if adapter_dir is not None and os.path.exists(os.path.join(adapter_dir, "critic.pth")):
            critic_params = torch.load(os.path.join(adapter_dir, "critic.pth"))
            self.v_head_mlp1.weight.data = critic_params["v_head_mlp1.weight"].to(self.v_head_mlp1.weight.device)
            self.v_head_mlp2.weight.data = critic_params["v_head_mlp2.weight"].to(self.v_head_mlp2.weight.device)
            self.v_head_mlp3.weight.data = critic_params["v_head_mlp3.weight"].to(self.v_head_mlp3.weight.device)
            cprint(f"[Critic] Loaded critic linear layer from {adapter_dir}", "green")

        self.pad_token_id = pad_token_id

    def gradient_checkpointing_enable(self, **kwargs):
        """
        Gradient checkpointing is for reducing memory usage.
        """
        self.rwtranrsformer.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self, **kwargs):
        self.rwtranrsformer.gradient_checkpointing_disable(**kwargs)

    def forward(
            self,
            input_ids, 
            attention_mask, 
            pixel_values,
            **kwargs,
        ):
        """
        input_ids: [B, L], e.g. [2, 292]
        attention_mask: [B, L], e.g. [2, 292]
        pixel_values: [B, 6, H, W], e.g. [2, 6, 224, 224]
        """
        # with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): 

        # print(f"{pixel_values.mean((1, 2, 3))=}")

        transformer_outputs = self.rwtranrsformer(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            pixel_values=pixel_values,    # [B, 6, H, W]
            output_hidden_states=True,
            **kwargs,
        )

        hidden_states = transformer_outputs.hidden_states[-1] # [B, L, D], e.g. [2, 292, 4096]
        # Option 1: use the last token
        # hidden_states = hidden_states[:, -1, :].float()  # [X, all the same]
        # Option 2: use the mean of all tokens
        text_features = hidden_states.mean(dim=1).float()   # [B, D], e.g. [2, 4096]

        # print(f"{text_features.mean(-1)=}")

        x = self.relu(self.v_head_mlp1(text_features))
        x = self.relu(self.v_head_mlp2(x))
        values = self.v_head_mlp3(x).squeeze(-1)
        return values   # [B,]
    
    def print_trainable_parameters(self):
        r"""
        Returns the number of trainable parameters and the number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        print(
            f"[Critic] trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
        )

    def save(self, save_dir):
        # save all parameters except the rwtranrsformer
        save_params = {}
        for name, param in self.named_parameters():
            if "rwtranrsformer" not in name:    # only save the linear layers
                save_params[name] = param.detach().cpu()

        torch.save(save_params, os.path.join(save_dir, "critic.pth"))


class CriticQwen(nn.Module):
    def __init__(self, all_args, adapter_dir=None):
        super().__init__()

        from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor  # NOTE: needs transformers >= 4.46.0 
        
        self.all_args = all_args
        self.prm_model_name_or_path = all_args.prm_model_name_or_path
        self.prm_checkpoint_path = all_args.prm_checkpoint_path
        print(f"prm_base_model_path: {self.prm_model_name_or_path}")
        print(f"prm_checkpoint_path: {self.prm_checkpoint_path}")

        self.processor = AutoProcessor.from_pretrained(
            self.prm_model_name_or_path,
            # do_rescale=False,
        )  #, add_eos_token=False, padding_side='left')

        self.good_text = "1"
        self.good_token = self.processor.tokenizer.encode(self.good_text, add_special_tokens=False)[0]
        self.bad_text = "0"
        self.bad_token = self.processor.tokenizer.encode(self.bad_text, add_special_tokens=False)[0]
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(self.prm_model_name_or_path, 
                                                        device_map="auto", 
                                                        torch_dtype=torch.bfloat16,
                                                        attn_implementation="flash_attention_2",
                                                        )
        if adapter_dir is not None:
            self.model = PeftModel.from_pretrained(self.model, adapter_dir, is_trainable=True)
        elif self.prm_checkpoint_path is not None:
            self.model = PeftModel.from_pretrained(self.model, self.prm_checkpoint_path, is_trainable=True)

        if is_peft_model(self.model):
            if not self.all_args.use_lora:
                self.model.merge_and_unload()
                print(f"Merged and unloaded PEFT model")

        # self.v_head_mlp1 = nn.Linear(self.model.config.hidden_size, 1024, bias=False)
        # self.v_head_mlp2 = nn.Linear(1024, 512, bias=False)
        # self.v_head_mlp3 = nn.Linear(512, 1, bias=False)
        self.v_head_mlp = nn.Linear(self.model.config.hidden_size, 1, bias=False)
        self.relu = nn.ReLU()

    def preprocess(self, text_list: list, image_list: list, actions=None) -> np.ndarray:
        new_text_list = []
        new_image_list = []
        for i in range(len(text_list)):
            # extract task instruction from f"In: What action should the robot take to {task_label.lower()}?\nOut:"
            task_label = text_list[i].split("What action should the robot take to ")[1].split("?\nOut:")[0]
            new_image_list.append(Image.fromarray(image_list[i]))
            # Sample messages for batch inference
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_list[i]},
                        {"type": "text", "text": f"The task is {task_label}, is it completed?"},
                    ],
                },
                {
                    "role": "assistant",
                    # "content": self.good_text,
                    "content": self.bad_text,
                }
            ]
            # Preparation for batch inference
            new_text_list.append(self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True))
        return new_text_list, new_image_list

    def forward(self, text_list: list, image_list: list) -> np.ndarray:
        text_list, image_list = self.preprocess(text_list, image_list)

        # image is [0, 255]
        inputs = self.processor(
            text=text_list,
            images=image_list,
            videos=None,
            padding=True,   # left padding
            return_tensors="pt",
        ).to("cuda", dtype=torch.bfloat16)
        transformer_outputs = self.model(   # just output 0 or 1
            **inputs,
        )   # odict_keys(['logits'], ['past_key_values', 'hidden_states', 'attentions', 'rope_deltas'])
        # logits: [B, L, V], e.g, [B, 109, 151936]
        logits = transformer_outputs.logits     # [B, L, V]

        # get the probability of the 'good token' as values
        token_indices = (inputs.input_ids == self.bad_token).nonzero(as_tuple=True)[1]    # [B,] e.g, [109, 110]
        probs = torch.softmax(logits, dim=-1)   # [B, L, V]
        values = probs[torch.arange(logits.size(0)), token_indices - 1, self.bad_token]

        return values

    def save(self, save_dir):
        self.model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self, **kwargs):
        self.model.gradient_checkpointing_disable(**kwargs)

class CriticFilm(nn.Module):
    def __init__(self, text_encoder):
        import torchvision.models as models
        super().__init__()
        # Visual encoder (pretrained ResNet-18)
        self.resnet = models.resnet18(pretrained=True)
        # Remove the final fully connected layer
        self.visual_encoder = nn.Sequential(*list(self.resnet.children())[:-2])
        self.visual_feature_dim = 512
        
        # Text encoder
        self.text_encoder = text_encoder
        self.text_feature_dim = 768
        
        # FiLM generation layers with layer norm
        self.film_generator = nn.Sequential(
            nn.Linear(self.text_feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 2 * self.visual_feature_dim)  # 2x for gamma and beta
        )
        
        # Layer norm for feature modulation
        self.layer_norm = nn.LayerNorm([self.visual_feature_dim, 7, 7])  # Assuming ResNet output size
        
        # Final layers for value prediction with layer norm
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.visual_feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Freeze BERT and ResNet parameters to use pretrained weights
        for param in self.bert.parameters():
            param.requires_grad = False
        
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

    def apply_film(self, visual_features, film_params):
        """Apply FiLM conditioning to visual features"""
        # Split film_params into gamma (scale) and beta (shift)
        gamma, beta = torch.split(film_params, self.visual_feature_dim, dim=1)
        
        # Reshape for broadcasting
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        
        modulated_features = gamma * visual_features + beta
        modulated_features = self.layer_norm(modulated_features)
        
        return modulated_features

    def forward(self, input_ids: torch.Tensor, pixel_values: torch.Tensor, attention_mask: torch.Tensor=None) -> torch.Tensor:
        """
        Forward pass of the FiLM critic
        
        Args:
            input_ids: Text input tensor [batch_size, sequence_length] 
               (token IDs from BERT tokenizer)
            pixel_values: Visual input tensor [batch_size, channels, height, width]

        Returns:
            Value estimations [batch_size, 1]
        """
        # Extract visual features
        with torch.no_grad():
            visual_features = self.visual_encoder(pixel_values) # [B, 512, 7, 7]
        # Extract text features from BERT (using only the [CLS] token embedding)
        with torch.no_grad():
            text_outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
            # Use pooler_output which is safer than trying to extract the CLS token
            # This is the representation of the [CLS] token through a linear layer and tanh
            # text_features = text_outputs.pooler_output
            text_features = text_outputs.last_hidden_state[:, 0]
        
        film_params = self.film_generator(text_features)
        modulated_features = self.apply_film(visual_features, film_params)
        value = self.value_head(modulated_features)
        return value

    @torch.no_grad()
    def print_trainable_parameters(self):
        """
        Returns the number of trainable parameters and the number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        print(f"[Critic] trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}")


if __name__ == "__main__":
    from transformers import AutoModel
    text_encoder = AutoModel.from_pretrained(
        "distilbert/distilbert-base-uncased", 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2",
    )
    critic = CriticFilm(text_encoder).to(device=torch.device("cuda"))
    critic.to(dtype=torch.bfloat16)
    critic.print_trainable_parameters()

    input_ids = torch.randint(0, 100, (1, 10)).to(device=torch.device("cuda"), dtype=torch.long)
    pixel_values = torch.randn(1, 3, 224, 224).to(device=torch.device("cuda"), dtype=torch.float32)
    attention_mask = input_ids != 0

    with torch.autocast("cuda", dtype=torch.bfloat16):
        print(f"{input_ids.dtype=}, {pixel_values.dtype=}")
        value = critic(input_ids, pixel_values, attention_mask=attention_mask)
    print(value)
