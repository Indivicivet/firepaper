from dataclasses import dataclass, replace

import numpy as np
from PIL import Image
from scipy import signal


@dataclass
class PaperState:
    temp: np.ndarray
    char: np.ndarray = None
    wet: np.ndarray = None

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
    return replace(
        paper,
        temp=new_temp,
    )


if __name__ == "__main__":
    from pathlib import Path

    OUT_PATH = Path(__file__).parent / "working_outputs"
    OUT_PATH.mkdir(exist_ok=True, parents=True)

    paper = PaperState.blank(512)
    paper.temp[250:262, 250:262] = 1
    for i in range(100):
        if i % 10 == 0:
            paper.render_channels().save(OUT_PATH / f"tick{i}.png")
        paper = tick(paper)
