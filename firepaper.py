from dataclasses import dataclass, replace

import numpy as np
from PIL import Image
from scipy import signal


@dataclass
class PaperState:
    temp: np.ndarray
    char: np.ndarray = None
    wet: np.ndarray = None
    ignited: np.ndarray = None

    @classmethod
    def blank(cls, width=512, height=None):
        if height is None:
            height = width
        return cls(temp=np.zeros((width, height)))

    @classmethod
    def from_rgb_channels(cls, rgb_image):
        if not isinstance(rgb_image, np.ndarray):
            rgb_image = np.array(rgb_image)
        rgb_image = rgb_image.astype("float32") / rgb_image.max()
        return cls(
            temp=rgb_image[..., 0],
            char=rgb_image[..., 1],
            wet=rgb_image[..., 2],
        )

    def __post_init__(self):
        if self.char is None:
            self.char = np.zeros_like(self.temp)
        if self.wet is None:
            self.wet = np.zeros_like(self.temp)
        if self.ignited is None:
            self.ignited = np.zeros_like(self.temp)

    def render_channels(self):
        return Image.fromarray(
            np.clip(
                255 * np.dstack([
                    self.temp,  # red
                    self.char,  # green
                    self.wet,  # blue
                ]),
                0,
                255,
            ).astype("uint8")
        )

    def render_image(self):
        # todo :: obvs could do way more sensible compositing here
        im = np.full((*self.temp.shape, 3), 255.0)
        im[..., 1] -= self.temp * 255  # red
        im[..., 2] -= self.temp * 255  # red
        im[..., 0] -= self.wet * 255  # blue
        im[..., 1] -= self.wet * 255  # blue
        im *= np.atleast_3d(1 - self.char)
        return Image.fromarray(im.astype("uint8"))


def tick(
    paper: PaperState,
    # todo :: tick param type thing?
    ignition_temp_dry: float = 0.3,
    ignition_temp_wet: float = 1.2,
    # if > 1, fully wet paper cannot burn while heat is capped at 1
    # todo :: define a heat_cap ?
) -> PaperState:
    # temporary temp propagation :)
    # todo :: diffusion equation
    # todo :: effects of heat need to be much more nonlinear, I imagine
    temp_prop_kernel = np.ones((5, 5))
    temp_prop_kernel[1:4, 1:4] = 2.3  # truly excellent approximation to a gaussian :)
    temp_prop_kernel[2, 2] = 3
    temp_prop_kernel /= np.sum(temp_prop_kernel)
    new_temp = signal.convolve(
        paper.temp,
        temp_prop_kernel,
        mode="same",
    )

    # ambient heat loss
    new_temp *= 0.98

    new_temp += 0.05 * paper.ignited
    new_temp = np.minimum(new_temp, 1)

    new_ignited = np.logical_and(
        np.logical_or(
            paper.ignited,
            # new ignition conditions:
            paper.temp > (
                # todo :: looks a bit dodgy?
                ignition_temp_dry + (ignition_temp_wet - ignition_temp_dry) * paper.wet
            )
        ),
        paper.char < 0.99,
    )

    # todo :: more interesting things...
    new_char = paper.char.copy()
    new_char += 0.3 * paper.ignited * np.maximum(0, paper.temp - 0.5)
    new_char = np.minimum(new_char, 1)  # max charred = 1
    # todo :: could clip values to 0,1 in post_init?
    # (don't clip temp, shouldn't saturate)

    # todo :: also diffusion equation for wetness!
    wet_prop_kernel = np.ones((3, 3))
    wet_prop_kernel[1, 1] = 4
    wet_prop_kernel /= np.sum(wet_prop_kernel)
    new_wet = signal.convolve(
        paper.wet,
        wet_prop_kernel,
        mode="same",
    )
    new_wet -= paper.temp * 0.03
    new_wet = np.maximum(0, new_wet)

    return replace(
        paper,
        temp=new_temp,
        char=new_char,
        wet=new_wet,
        ignited=new_ignited,
    )


if __name__ == "__main__":
    from pathlib import Path

    OUT_PATH = Path(__file__).parent / "working_outputs"
    OUT_PATH.mkdir(exist_ok=True, parents=True)

    paper = PaperState.from_rgb_channels(Image.open("data_sources/example_start_1.png"))
    for i in range(201):
        if i % 10 == 0:
            debug_file = OUT_PATH / f"debug{i}.png"
            image_file = OUT_PATH / f"image{i}.png"
            print(f"exporting to {debug_file} and {image_file}")
            paper.render_channels().save(debug_file)
            paper.render_image().save(image_file)
        paper = tick(paper)
