# Plot slices of the array and save to a gif
import os

import imageio
import matplotlib.pyplot as plt
from tqdm import trange


def save_video(
  fields,
  filename: str,
  fps=30,
  vmin=None,
  vmax=None,
  cmap='RdBu_r',
  aspect='equal'
):
  r'''Saves a video of the fields to an mp4 file. Unix only.'''
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
