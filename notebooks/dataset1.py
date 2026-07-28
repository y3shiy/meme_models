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
    display_image(dataset1.get_image(index=5), width=200, markings=(100, 100), interactive=True)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Trial and error
    """)
    return


@app.cell(disabled=True)
def _():
    dataset_dir = kagglehub.dataset_download('hammadjavaid/6992-labeled-meme-images-dataset')
    dataset_dir = Path(dataset_dir)
    return (dataset_dir,)


@app.cell
def _(dataset_dir):
    df = pd.read_csv(dataset_dir/'labels.csv')
    df.head()
    return (df,)


@app.cell
def _():
    random_images_number = mo.ui.slider(1, 15, value=10, label='Number of random images')
    random_images_number
    return (random_images_number,)


@app.cell(disabled=True)
def _(dataset_dir, df, random_images_number):
    random.seed(42)

    images_dir = dataset_dir/'images'/'images'

    random_images_indices = random.sample(range(df.shape[0]), random_images_number.value)

    df_image_names = df.iloc[random_images_indices].image_name
    image_paths = [images_dir/image_path for image_path in df_image_names]

    image_widgets = []
    for image_path in df_image_names:
        with Image.open(images_dir/image_path) as image:
            image.thumbnail((200, 200))
            image_widgets.append(mo.image(image))

    mo.vstack([
        mo.hstack(widgets) for widgets in batched(image_widgets, 5)
    ])
    return


if __name__ == "__main__":
    app.run()
