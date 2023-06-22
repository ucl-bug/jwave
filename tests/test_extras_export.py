import os

import numpy as np
import pytest
from jaxdf import Domain, OnGrid

from jwave.extras import save_video

# Create mock field
domain = Domain((
    64,
    64,
), (
    1,
    1,
))
field_params = np.random.rand(5, *domain.N, 1)
fields = OnGrid(field_params, domain)

filenames = [
    'video1.mp4',
]
fpss = [
    30,
]
vmins = [
    1,
]
vmaxs = [
    2,
]
cmaps = [
    'RdBu_r',
]
aspects = ['equal', 'auto']


@pytest.mark.parametrize('filename', filenames)
@pytest.mark.parametrize('fps', fpss)
@pytest.mark.parametrize('vmin', vmins)
@pytest.mark.parametrize('vmax', vmaxs)
@pytest.mark.parametrize('cmap', cmaps)
@pytest.mark.parametrize('aspect', aspects)
def test_save_video(filename, fps, vmin, vmax, cmap, aspect):
    save_video(fields, filename, fps, vmin, vmax, cmap, aspect)
    assert os.path.exists(filename)    # check if the file was created
    assert filename.endswith('.mp4')    # check if the file is an mp4 file
    os.remove(filename)    # remove the file after the test

    # Additional asserts can be added to test the video content.
    # This is more complex and requires video processing libraries
    # to analyze the video frames and properties.


pytest.main()
