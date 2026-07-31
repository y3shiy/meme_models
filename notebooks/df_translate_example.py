import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")

with app.setup:
    import os

    import marimo as mo
    import pandas as pd
    from df_translate import DeepL


@app.cell
async def _():
    deepl_api_key = os.environ.get('DEEPL_API_KEY')
    deepl = await DeepL.create(deepl_api_key, is_free_api=True)
    return (deepl,)


@app.cell
def _():
    df = pd.DataFrame({'label': ['Pillow','Kiwi','Fridge','Dog']})
    df
    return (df,)


@app.cell
async def _(deepl, df):
    translated_df = await deepl.translate_dataframe(df, 'en', ['ru','uk','pl','sl','fr','de'])
    translated_df
    return


if __name__ == "__main__":
    app.run()
