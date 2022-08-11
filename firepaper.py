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


def tick(
    paper: PaperState,
    # todo :: tick param type thing?
    ignition_temp_dry: float = 0.3,
    ignition_temp_wet: float = 0.9,
) -> PaperState:
    # temporary temp propagation :)
    # todo :: diffusion equation
    temp_prop_kernel = np.ones((5, 5))
    temp_prop_kernel[2, 2] = 3
    temp_prop_kernel /= np.sum(temp_prop_kernel)
    new_temp = signal.convolve(
        paper.temp,
        temp_prop_kernel,
        mode="same",
    )

    # ambient heat loss
    new_temp *= 0.99

    new_temp += 0.1 * paper.ignited
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

    paper = PaperState.blank(512)
    paper.temp[250:262, 250:262] = 1
    paper.wet[:, 230:240] = 1
    for i in range(201):
        if i % 10 == 0:
            out_file = OUT_PATH / f"tick{i}.png"
            print(f"exporting to {out_file}")
            paper.render_channels().save(out_file)
        paper = tick(paper)
