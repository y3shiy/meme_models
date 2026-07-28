import marimo

__generated_with = "0.23.15"
app = marimo.App(width="columns")

with app.setup:
    import easyocr
    import marimo as mo
    import numpy as np
    import matplotlib as plt
    from dataset1 import Dataset1
    from imageutils import plot_image, display_image, extract_text_fragments
    from matplotlib.patches import Polygon
    from PIL import Image


@app.cell
def _():
    dataset1 = Dataset1()
    dataset1.labels.head()
    return (dataset1,)


@app.cell
def _(dataset1):
    image = dataset1.get_image(index=3)
    display_image(image, width=800, ticks=(100, 100), interactive=True)
    return (image,)


@app.cell
def _(cropped_fragmets, image):
    image_fragments = extract_text_fragments(image, ['en'])
    mo.vstack([mo.image(image, width=300) for image in cropped_fragmets])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Trying to build the pipeline
    """)
    return


@app.cell
def _():
    reader = easyocr.Reader(['en'])
    return (reader,)


@app.cell
def _(image, reader):
    image_bytes = np.asarray(image.convert('RGB'))
    ocr_result = reader.readtext(image_bytes)
    ocr_result
    return (ocr_result,)


@app.cell
def _(image, ocr_result):
    fig, ax = plot_image(image)

    for region in ocr_result:
        confidence = region[2]
        if confidence > 0.3:
            rect = region[0]

            polygon = Polygon(
                rect,
                closed=True,
                edgecolor='red',
                facecolor='none',
                linewidth=2,
            )
            ax.add_patch(polygon)

    fig
    return


@app.cell(column=1, hide_code=True)
def _():
    mo.md(r"""
    #sing the OCR result
    """)
    return


@app.cell
def _(image, ocr_result):
    cropped_fragmets = []

    for region_info in ocr_result:
        rect_ = region_info[0]
        text_ = region_info[1]
        confidence_ = region_info[2]

        coords = (
            rect_[0][0],
            rect_[0][1],
            rect_[2][0],
            rect_[2][1]
        )

        if confidence_ > 0.3:
            cropped_image = image.crop(coords)
            cropped_fragmets.append(cropped_image)

    mo.vstack([mo.image(image, width=300) for image in cropped_fragmets])
    return (cropped_fragmets,)


if __name__ == "__main__":
    app.run()
