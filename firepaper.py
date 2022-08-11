from dataclasses import dataclass

import numpy as np
from PIL import Image


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


if __name__ == "__main__":
    paper = PaperState.blank()
    paper.render_channels().show()
