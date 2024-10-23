from torchvision import transforms

# Define normalization constants
NORMALIZATION_STATS = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "openai_clip": (
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711],
    ),
    "none": (None, None),
}


def get_normalization_stats(norm_type="imagenet"):
    try:
        return NORMALIZATION_STATS[norm_type]
    except KeyError:
        raise ValueError(f"Invalid normalization type: {norm_type}")


def create_eval_transforms(mean, std, img_size=-1, center_crop=False):
    transform_list = []

    if img_size > 0:
        transform_list.append(transforms.Resize(img_size))
    if center_crop:
        assert img_size > 0, "img_size must be set if center_crop is True"
        transform_list.append(transforms.CenterCrop(img_size))

    transform_list.append(transforms.ToTensor())
    if mean is not None and std is not None:
        transform_list.append(transforms.Normalize(mean, std))

    return transforms.Compose(transform_list)
