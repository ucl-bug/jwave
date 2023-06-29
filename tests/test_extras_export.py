import os

from jax import random
from jaxdf import Domain, OnGrid

from jwave.extras import save_video
from jwave.extras.export import save_video


def test_save_video():
    # Generate a test Field object
    key = random.PRNGKey(0)
    field_data = random.normal(key, (10, 64, 64, 1))
    domain = Domain((64, 64), (1, 1))
    field = OnGrid(field_data, domain)

    # Define the filename for the video
    filename = "test.mp4"

    # Ensure the file does not exist before the function call
    if os.path.exists(filename):
        os.remove(filename)

    # Call the function
    save_video(field, filename)

    # Check that the video file was created
    assert os.path.exists(filename), "Video file was not created"

    # Clean up by removing the created video file
    os.remove(filename)
