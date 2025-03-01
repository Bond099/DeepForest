"""A deepforest callback Callbacks must have the following methods
on_epoch_begin, on_epoch_end, on_fit_end, on_fit_begin methods and inject model
and epoch kwargs."""

from deepforest import visualize
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import glob
import tempfile

from pytorch_lightning import Callback
from deepforest import dataset
from deepforest import utilities
from deepforest import predict

import torch


class images_callback(Callback):
    """Run evaluation on a file of annotations during training.

    Args:
        savedir: optional, directory to save predicted images
        probability_threshold: minimum probablity for inclusion, see deepforest.evaluate
        n: number of images to upload
        select_random (False): whether to select random images or the first n images
        every_n_epochs: run epoch interval
        color: color of the bounding box as a tuple of BGR color, e.g. orange annotations is (0, 165, 255)
        thickness: thickness of the rectangle border line in px

    Returns:
        None: either prints validation scores or logs them to the pytorch-lightning logger
    """

    def __init__(self,
                 savedir,
                 n=2,
                 every_n_epochs=5,
                 select_random=False,
                 color=None,
                 thickness=1):
        self.savedir = savedir
        self.n = n
        self.color = color
        self.thickness = thickness
        self.select_random = select_random
        self.every_n_epochs = every_n_epochs

    def log_images(self, pl_module):
        # It is not clear if this is per device, or per batch. If per batch, then this will not work.
        df = pl_module.predictions[0]

        # limit to n images, potentially randomly selected
        if self.select_random:
            selected_images = np.random.choice(df.image_path.unique(), self.n)
        else:
            selected_images = df.image_path.unique()[:self.n]
        df = df[df.image_path.isin(selected_images)]

        # The new plot_results needs root_dir in the dataframe, not as an arg
        df["root_dir"] = pl_module.config["validation"]["root_dir"]
        if self.color:
            my_color = [self.color[2], self.color[1], self.color[0]]  # Flip it
        else:
            my_color = [245, 135, 66]  # Default orange from plot_results


        visualize.plot_results(
            results=df,  # Here’s the predictions
            savedir=self.savedir,  # Dump the pics here
            results_color=my_color,  # Use the flipped RGB color
            thickness=self.thickness  # Same old thickness
        )

        # Now let’s try to log these to whatever tracker we’re using
        try:
            saved_pics = glob.glob(f"{self.savedir}/*.png")
            for pic in saved_pics:
                pl_module.logger.experiment.log_image(pic)
        except Exception as e:
            print(f"Eh, logging didn’t work—saved to {self.savedir} anyway. Error was: {e}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:  # optional skip
            return

        if trainer.current_epoch % self.every_n_epochs == 0:
            print("Running image callback")
            self.log_images(pl_module)
