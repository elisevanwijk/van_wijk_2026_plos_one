from pathlib import Path

import pandas as pd

data_folder = Path("./data")
intermediate_folder = Path("./intermediate")
results_folder = Path("./results")

results_folder.mkdir(exist_ok=True)


def write_excel(dfs, path, autoformat=True):
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        for sheetname, df_ in dfs.items():  # loop through `dict` of dataframes
            df_.to_excel(writer, sheet_name=sheetname)  # send df to writer
            if autoformat:
                worksheet = writer.sheets[sheetname]  # pull worksheet object
                df_no_index = df_.reset_index()
                for idx, col in enumerate(df_no_index):  # loop through all columns
                    series = df_no_index[col]
                    max_len = (
                        max(
                            (
                                series.astype(str)
                                .map(len)
                                .max(),  # len of largest item
                                len(str(series.name)),  # len of column name/header
                            )
                        )
                        + 1
                    )  # adding a little extra space
                    worksheet.set_column(idx, idx, max_len)  # set column width
