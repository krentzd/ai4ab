"""OverlappingCropMultiChannel: tile a (C, H, W) tensor into square crops."""

import math

import numpy as np
import torch


class OverlappingCropMultiChannel:
    """Raster-scan a (C, H, W) tensor into square crops, discarding boundary remainders. pad=True zero-pads first."""

    def __init__(self, crop_size, stride, pad: bool = True, mode: str = 'RGB') -> None:
        assert isinstance(crop_size, (int, tuple))
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            assert len(crop_size) == 2
            self.crop_size = crop_size
        self.stride = stride
        self.mode = mode
        self.pad = pad

    def __call__(self, image: torch.Tensor):
        ch, h, w = image.size()
        new_h, new_w = self.crop_size

        if self.pad and (h % new_h != 0 or w % new_w != 0):
            old_h = h
            old_w = w
            h = math.ceil(h / new_h) * new_h
            w = math.ceil(w / new_w) * new_w

            pad_h = h - old_h
            pad_w = w - old_w
            pad_h_top = math.floor(pad_h / 2)
            pad_h_bottom = math.ceil(pad_h / 2)
            pad_w_left = math.floor(pad_w / 2)
            pad_w_right = math.ceil(pad_w / 2)

            image = np.stack(
                [
                    np.pad(
                        image[i, :, :],
                        ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)),
                        'constant',
                        constant_values=0,
                    )
                    for i in range(ch)
                ],
                axis=2,
            )

        image_crops = [
            image[:, i:i + new_h, j:j + new_w]
            for i in range(0, h, self.stride)
            for j in range(0, w, self.stride)
        ]
        image_crops = [crop for crop in image_crops if crop.size()[1:] == (new_h, new_w)]

        return image_crops
