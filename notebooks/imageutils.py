import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")

with app.setup:
    import easyocr
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from anywidget import AnyWidget
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@app.function
def plot_image(image: Image,
               ticks: tuple[int, int] | None = None) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots()
    
    ax.imshow(image, origin='upper')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    if ticks is not None:
        ax.set_xticks(range(0, image.width, ticks[0]))
        ax.set_yticks(range(0, image.height, ticks[1]))
        ax.grid()
        
    return fig, ax


@app.function
def display_image(image: Image,
                  width: int | None = None,
                  height: int | None = None,
                  ticks: tuple[int, int] | None = None,
                  interactive: bool = False) -> AnyWidget:
    has_ticks = ticks is not None
    
    match (has_ticks, interactive):
        case False, False:
            return mo.image(image, width=width, height=height)
        case False, True: 
            raise NotImplementedError()
        case True, _:
            assert len(ticks) == 2
            assert ticks[0] > 0
            assert ticks[1] > 0

            fig, ax = plot_image(image, ticks)

            if interactive:
                interactive_plot = mo.mpl.interactive(fig)
                return interactive_plot
            else:
                styles = []
                if width is not None:
                    styles.append(f'width: {width}px')
                if height is not None:
                    styles.append(f'height: {height}px')
                    
                return mo.Html(
                    f'''
                    <div style="{';'.join(styles)}">
                    {mo.as_html(fig)}
                    </div>
                    '''
                )


@app.function
def extract_text_fragments(image: Image, languages: list[str]) -> list[Image]:
    reader = easyocr.Reader(languages)

    image_bytes = np.asarray(image.convert('RGB'))
    ocr_result = reader.readtext(image_bytes)

    cropped_fragments = []
    for region_info in ocr_result:
        rect, text, confidence = region_info

        crop_coords = (rect[0][0], rect[0][1],
                       rect[2][0], rect[2][1])

        if confidence >= 0.3:
            cropped_image = image.crop(crop_coords)
            cropped_fragments.append(cropped_image)

    return cropped_fragments


if __name__ == "__main__":
    app.run()
