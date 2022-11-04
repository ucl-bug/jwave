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

import os

import imageio
import matplotlib.pyplot as plt
from jaxdf import Field
from tqdm import trange


def save_video(
  fields: Field,
  filename: str,
  fps=30,
  vmin=None,
  vmax=None,
  cmap='RdBu_r',
  aspect='equal'
):
  r'''Saves a video of the fields to an mp4 file. **Unix only**. It
  requires ffmpeg and the package `imageio` to be installed.

  Args:
    fields (Field): The fields to save. It needs to be a Field object
      with a time dimension, represented as a leading index. It must be
      therefore possible to iterate over the time dimension using
      ``for field in fields``, or generally being able to access the
      fields as ``fields[i]``.
    filename (str): The name of the file to save the video to.
    fps (int, optional): The number of frames per second. Defaults to 30.
    vmin (float, optional): The minimum value of the color scale. If None,
      the minimum value of the fields at each time step is used. Defaults
      to None.
    vmax (float, optional): The maximum value of the color scale. If None,
      the maximum value of the fields at each time step is used. Defaults
      to None.
    cmap (str, optional): The matplotlib colormap to use. Defaults to
      'RdBu_r'.
    aspect (str, optional): The aspect ratio of the plot. Defaults to
      'equal'.
  '''
  # Make a temporary directory in /tmp
  tmp_dir = os.path.join('/tmp', 'jwave_video')

  # Clean the directory if it exists
  if os.path.exists(tmp_dir):
    os.system(f'rm -rf {tmp_dir}')
  os.mkdir(tmp_dir)

  # Saves all images in the temporary director
  for i in trange(fields.params.shape[0]):
      img_data = fields[i].on_grid
      plt.imshow(img_data, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect)
      plt.colorbar()
      # Save with leading zeros
      plt.savefig(f'{tmp_dir}/frame_{i:08}.png')
      plt.close()

  # Create the video
  writer = imageio.get_writer(filename, fps=fps)
  for filename in sorted(os.listdir(tmp_dir)):
    frame = imageio.imread(os.path.join(tmp_dir, filename))
    writer.append_data(frame)
  writer.close()

  # Clean up
  os.system(f'rm -rf {tmp_dir}')
