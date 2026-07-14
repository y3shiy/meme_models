import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from df_translate import DeepL

    return (DeepL,)


@app.cell
async def _(DeepL):
    import os
    deepl_api_key = os.environ.get('DEEPL_API_KEY')
    deepl = await DeepL.create(deepl_api_key, is_free_api=True)
    return (deepl,)


@app.cell
def _():
    import pandas as pd
    df = pd.DataFrame({'label': ['Cat', 'Frog']})
    df
    return (df,)


@app.cell
async def _(deepl, df):
    translated_df = await deepl.translate_dataframe(df, 'en', ['ru','uk','pl','sl','fr','de'])
    translated_df
    return


if __name__ == "__main__":
    app.run()
