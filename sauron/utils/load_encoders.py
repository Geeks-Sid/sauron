import traceback
from abc import ABC, abstractmethod

import torch
from torchvision import transforms

from sauron.utils.transform_utils import create_eval_transforms, get_normalization_stats


class BaseInferenceModel(torch.nn.Module, ABC):
    def __init__(self, weights_path=None, **build_kwargs):
        super().__init__()
        self.weights_path = weights_path
        self.model, self.eval_transforms, self.precision = self._build_model(
            weights_path, **build_kwargs
        )

    @abstractmethod
    def _build_model(self, weights_path, **build_kwargs):
        pass

    def forward(self, x):
        return self.model(x)


class CustomEncoder(BaseInferenceModel):
    def __init__(self, model, transforms, precision, weights_path=None):
        self.model = model
        self.eval_transforms = transforms
        self.precision = precision
        super().__init__(weights_path)

    def _build_model(self, weights_path, **kwargs):
        return self.model, self.eval_transforms, self.precision


class ConchEncoder(BaseInferenceModel):
    def _build_model(self, weights_path=None, **kwargs):
        try:
            from conch.open_clip_custom import create_model_from_pretrained
        except ImportError as e:
            traceback.print_exc()
            raise ImportError(
                "Please install CONCH via `pip install git+https://github.com/Mahmoodlab/CONCH.git`"
            ) from e

        try:
            model, preprocess = create_model_from_pretrained(
                "conch_ViT-B-16", "hf_hub:MahmoodLab/conch"
            )
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(
                "Failed to download CONCH model. Ensure you have access and your token is correctly registered."
            ) from e

        return model, preprocess, torch.float32

    def forward(self, x):
        return self.model.encode_image(x, proj_contrast=False, normalize=False)


class CTransPathEncoder(BaseInferenceModel):
    def _build_model(self, weights_path, **kwargs):
        from torch import nn

        from .ctranspath.ctran import ctranspath

        model = ctranspath(img_size=224)
        model.head = nn.Identity()

        state_dict = torch.load(weights_path)["model"]
        state_dict = {k: v for k, v in state_dict.items() if "attn_mask" not in k}
        model.load_state_dict(state_dict, strict=False)

        mean, std = get_normalization_stats("imagenet")
        eval_transforms = create_eval_transforms(mean, std)
        return model, eval_transforms, torch.float32


class HOptimus0Encoder(BaseInferenceModel):
    def _build_model(self, weights_path=None, **kwargs):
        import timm

        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
            **kwargs,
        )

        eval_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )

        return model, eval_transforms, torch.float16


class PhikonEncoder(BaseInferenceModel):
    def _build_model(self, weights_path=None, **kwargs):
        from transformers import ViTModel

        model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        mean, std = get_normalization_stats("imagenet")
        eval_transforms = create_eval_transforms(mean, std)
        return model, eval_transforms, torch.float32

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.last_hidden_state[:, 0, :]


class PlipEncoder(BaseInferenceModel):
    def _build_model(self, weights_path=None, **kwargs):
        from transformers import CLIPImageProcessor, CLIPVisionModel

        model_name = "vinid/plip"
        image_processor = CLIPImageProcessor.from_pretrained(model_name)
        model = CLIPVisionModel.from_pretrained(model_name)

        def eval_transform(img):
            return image_processor(img, return_tensors="pt")["pixel_values"].squeeze(0)

        return model, eval_transform, torch.float32

    def forward(self, x):
        return self.model(x).pooler_output


class RemedisEncoder(BaseInferenceModel):
    def _build_model(self, weights_path, **kwargs):
        from .remedis.remedis_models import resnet152_remedis

        model = resnet152_remedis(ckpt_path=weights_path, pretrained=True)
        return model, None, torch.float32

    def forward(self, x):
        x = x.permute((0, 3, 1, 2))
        return self.model(x)


class ResNet50Encoder(BaseInferenceModel):
    def _build_model(self, weights_path=None, pretrained=True, pool=True, **kwargs):
        import timm

        model = timm.create_model(
            "resnet50.tv_in1k",
            pretrained=pretrained,
            features_only=True,
            out_indices=[3],
            num_classes=0,
            **kwargs,
        )
        mean, std = get_normalization_stats("imagenet")
        eval_transforms = create_eval_transforms(mean, std)
        self.pool = torch.nn.AdaptiveAvgPool2d(1) if pool else None
        return model, eval_transforms, torch.float32

    def forward(self, x):
        features = self.model(x)
        if isinstance(features, list):
            features = features[0]
        if self.pool:
            features = self.pool(features).squeeze(-1).squeeze(-1)
        return features


class UniEncoder(BaseInferenceModel):
    def _build_model(self, weights_path=None, **kwargs):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        model = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            dynamic_img_size=True,
            num_classes=0,
            init_values=1.0,
            **kwargs,
        )

        config = resolve_data_config(model.pretrained_cfg, model=model)
        eval_transforms = create_transform(**config)
        return model, eval_transforms, torch.float16


class GigaPathEncoder(BaseInferenceModel):
    def _build_model(self, weights_path=None, **kwargs):
        import timm

        model = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", pretrained=True, **kwargs
        )

        eval_transforms = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        return model, eval_transforms, torch.float32


class VirchowEncoder(BaseInferenceModel):
    def _build_model(self, weights_path=None, return_cls=False, **kwargs):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        model = timm.create_model(
            "hf-hub:paige-ai/Virchow",
            pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            **kwargs,
        )
        config = resolve_data_config(model.pretrained_cfg, model=model)
        eval_transforms = create_transform(**config)
        self.return_cls = return_cls
        return model, eval_transforms, torch.float32

    def forward(self, x):
        output = self.model(x)
        class_token = output[:, 0]
        if self.return_cls:
            return class_token
        else:
            patch_tokens = output[:, 1:]
            embeddings = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)
            return embeddings


class Virchow2Encoder(BaseInferenceModel):
    def _build_model(self, weights_path=None, return_cls=False, **kwargs):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            **kwargs,
        )
        config = resolve_data_config(model.pretrained_cfg, model=model)
        eval_transforms = create_transform(**config)
        self.return_cls = return_cls
        return model, eval_transforms, torch.float16

    def forward(self, x):
        output = self.model(x)
        class_token = output[:, 0]
        if self.return_cls:
            return class_token
        else:
            patch_tokens = output[:, 5:]
            embedding = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)
            return embedding


ENCODER_REGISTRY = {
    "remedis": RemedisEncoder,
    "resnet50": ResNet50Encoder,
    "gigapath": GigaPathEncoder,
    "virchow": VirchowEncoder,
    "virchow2": Virchow2Encoder,
    "hoptimus0": HOptimus0Encoder,
    "conch_v1": ConchEncoder,
    "uni_v1": UniEncoder,
    "ctranspath": CTransPathEncoder,
    "phikon": PhikonEncoder,
    "plip": PlipEncoder,
}


def get_encoder_class(encoder_name):
    try:
        return ENCODER_REGISTRY[encoder_name]
    except KeyError:
        raise ValueError(f"Unknown encoder name {encoder_name}")
