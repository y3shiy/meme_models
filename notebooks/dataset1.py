import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")

with app.setup:
    import random
    from itertools import batched
    from pathlib import Path
    from typing import Any

    import kagglehub
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image
    from kagglehub import KaggleDatasetAdapter
    from pandas import DataFrame
    from imageutils import display_image


@app.class_definition
class Dataset1:
    def __init__(self):
        dataset_dir = kagglehub.dataset_download('hammadjavaid/6992-labeled-meme-images-dataset')
        self._dataset_dir = Path(dataset_dir)
        self._labels = pd.read_csv(self._dataset_dir/'labels.csv')

    @property
    def dataset_dir(self) -> Path:
        return self._dataset_dir

    @property
    def labels(self) -> DataFrame:
        return self._labels

    def get_image(self, image_name: str | None = None, index: int = 0) -> Image:
        if image_name is None:
            image_name = self.labels.iloc[index].image_name

        images_dir = self.dataset_dir/'images'/'images'
        with Image.open(images_dir/image_name) as image:
            image.load()
            return image

    def random_sample_labels(self, number: int, seed: int | None = None) -> DataFrame:
        if seed is not None:
            random.seed(seed)
        
        random_indices = random.sample(range(self.labels.shape[0]), number)
        return self.labels.iloc[random_indices]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Usage examples
    """)
    return


@app.cell
def _():
    dataset1 = Dataset1()
    dataset1.labels.head()
    return (dataset1,)


@app.cell
def _(dataset1):
    display_image(dataset1.get_image(index=5), width=200, ticks=(100, 100), interactive=True)
    return


@app.cell
def _():
    images_number = mo.ui.slider(1, 10, value=5, label='Number of random images')
    images_number
    return (images_number,)


@app.cell
def _(dataset1, images_number):
    sample = dataset1.random_sample_labels(images_number.value, seed=42)
    mo.hstack([display_image(dataset1.get_image(image_name), width=200) for image_name in sample.image_name])
    return


if __name__ == "__main__":
    app.run()
