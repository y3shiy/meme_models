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


@app.function
def display_image(image: Image,
                  width: int | None = None,
                  height: int | None = None,
                  markings: tuple[int, int] | None = None,
                  interactive: bool = False):
    width = width if width is not None else image.width
    height = height if height is not None else image.height
    has_markings = markings is not None

    match (has_markings, interactive):
        case False, False:
            return mo.image(image, width=width, height=height)
        case False, True: 
            raise NotImplementedError()
        case True, _:
            assert len(markings) == 2
            assert markings[0] > 0
            assert markings[1] > 0
            
            fig, ax = plt.subplots()
            ax.imshow(image, origin='upper')
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')
            ax.set_xticks(range(0, image.width, markings[0]))
            ax.set_yticks(range(0, image.height, markings[1]))
            ax.grid()

            if interactive:
                interactive_plot = mo.mpl.interactive(fig)
                return interactive_plot
            else:
                return mo.Html(
                    f'''
                    <div style='width: {width}px;'>
                    {mo.as_html(fig)}
                    </div>
                    '''
                )


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

    def get_image(self, image_name: str) -> Image:
        images_dir = self.dataset_dir/'images'/'images'
        with Image.open(images_dir/image_name) as image:
            image.load()
            return image

    def display_image(self, image_name: str, *args, **kwargs) -> Any:
        return display_image(self.get_image(image_name), *args, **kwargs)


@app.cell
def _():
    dataset1 = Dataset1()
    dataset1.labels.head()
    return (dataset1,)


@app.cell
def _(dataset1):
    image1_name = dataset1.labels.iloc[0].image_name
    image1_name
    return (image1_name,)


@app.cell
def _(dataset1, image1_name):
    dataset1.display_image(image1_name, markings=(100, 50), interactive=True)
    return


@app.cell(disabled=True)
def _():
    dataset_dir = kagglehub.dataset_download('hammadjavaid/6992-labeled-meme-images-dataset')
    dataset_dir = Path(dataset_dir)
    return (dataset_dir,)


@app.cell
def _(dataset_dir):
    for path in dataset_dir.iterdir():
        print(path)

    for i, path in enumerate((dataset_dir/'images'/'images').iterdir()):
        print(path)
        if i > 10:
            break
    return


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
