# This file is part of j-Wave.
#
# j-Wave is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
#
# j-Wave is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with j-Wave. If not, see <https://www.gnu.org/licenses/>.
from typing import Union

import numpy as np
from jaxdf import Field
from matplotlib import colormaps, colors


def save_video(
    fields: Field,
    filename: str,
    fps: int = 30,
    vmin: Union[None, float] = None,
    vmax: Union[None, float] = None,
    cmap="RdBu_r",
    aspect="equal",
    codec="mp4v",
):
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "Please install opencv-python to use the save_video function.")

    try:
        from tqdm import trange
    except ImportError:
        # Define a fallback function if tqdm is not available.
        def trange(*args, **kwargs):
            return range(*args, **kwargs)

    # Define the colormap
    cmap = colormaps[cmap]

    # Define a video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)

    # Assuming the field shape is known
    frame_height, frame_width, _ = fields[0].on_grid.shape
    writer = cv2.VideoWriter(filename, fourcc, fps,
                             (frame_width, frame_height))

    for i in trange(fields.params.shape[0]):
        img_data = fields[i].on_grid[:, :, 0]
        norm = colors.Normalize(vmin=vmin,
                                vmax=vmax) if vmin and vmax else None
        img_data = cmap(norm(img_data)) if norm else cmap(img_data)

        # Convert from RGBA to BGR and ignore the alpha channel
        img_data = cv2.cvtColor((img_data[:, :, :3] * 255).astype(np.uint8),
                                cv2.COLOR_RGBA2BGR)

        # If aspect ratio is not 'equal', resize the image according to the given aspect ratio
        if aspect != 'equal':
            aspect_ratio = tuple(map(int, aspect.split(':')))
            img_data = cv2.resize(img_data, aspect_ratio)

        writer.write(img_data)

    writer.release()
