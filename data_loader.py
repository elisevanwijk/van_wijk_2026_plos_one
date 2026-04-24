from functools import cache

import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


from common import intermediate_folder, results_folder, write_excel


def extract_quintile(scores, label, year):
    labels = ["poor", "average1", "average2", "average3", "excellent"]

    gpa_col = f"gpa_{label}_{year}"
    quintile_column = f"{label}_{year}_quintile"
    scores[quintile_column] = pd.qcut(
        scores[gpa_col],
        5,
        labels=labels,
    )

    scores[quintile_column] = scores[quintile_column].str.strip(r"[123]")
    scores[f"{label}_is_poor_quintile"] = (scores[quintile_column] == "poor").astype(
        "int"
    )
    scores[f"{label}_is_excellent_quintile"] = (
        scores[quintile_column] == "excellent"
    ).astype("int")
    scores[f"{label}_is_poor_gpa"] = (scores[gpa_col] < 6.0).astype("int")
    scores[f"{label}_is_excellent_gpa"] = (scores[gpa_col] >= 8.0).astype("int")


class DataLoader:
    @property
    @cache
    def gpa(self):
        return pd.read_excel(intermediate_folder / "gpa.xlsx").set_index(
            "student_nummer"
        )

    first_year_sheets = ["RM 2021", "RM 2022", "RM 2023"]
    combined_year_sheets = ["RM 2021_DA 2022", "RM 2022_DA 2023", "RM 2023_DA 2024"]

    @cache
    def _load_sheet(self, sheet_name):
        sheet = pd.read_excel(
            intermediate_folder / "student_scores_exams.xlsx", sheet_name=sheet_name
        ).set_index("Studentnummer")

        exam_takers = len(sheet)
        sheet = sheet.join(self.gpa, how="inner")

        gpa_available = len(sheet)

        if sheet_name in self.first_year_sheets:
            sheet = sheet.loc[sheet.ects_year_1 > sheet.ects_year_1.max() * 0.75]
        elif sheet_name in self.combined_year_sheets:
            sheet = sheet.loc[sheet.ects_total > sheet.ects_total.max() * 0.75]

        enough_ects = len(sheet)

        print(
            f"Loading {sheet_name}, ejected {exam_takers - gpa_available} for gpa availability, {gpa_available - enough_ects} for having 75% of ects available, resulting in {enough_ects}"
        )

        score_columns = [col for col in sheet.columns if col.startswith("score")]

        for col in score_columns:
            mean = sheet[col].mean()
            std = sheet[col].std()
            sheet["z_" + col] = (sheet[col] - mean) / std

        return sheet

    def rm_per_cohort(self):
        for sheet_name in self.first_year_sheets:
            sheet = self._load_sheet(sheet_name)

            extract_quintile(sheet, "last_attempt", "year_1")
            extract_quintile(sheet, "first_attempt", "year_1")
            yield sheet_name, sheet

    def rm_dm_per_cohort(self):
        for sheet_name in self.combined_year_sheets:
            sheet = self._load_sheet(sheet_name)

            extract_quintile(sheet, "last_attempt", "total")
            extract_quintile(sheet, "first_attempt", "total")
            yield sheet_name, sheet

    def rm_total(self):
        return pd.concat([x for _, x in self.rm_per_cohort()])

    def rm_dm_total(self):
        return pd.concat([x for _, x in self.rm_dm_per_cohort()])

    def generate_result_sheets(self):
        dfs = {}
        for sheet_name, data in self.rm_per_cohort():
            dfs[sheet_name] = data
        dfs["RM Total"] = self.rm_total()
        for sheet_name, data in self.rm_dm_per_cohort():
            dfs[sheet_name] = data
        dfs["RM DM Total"] = self.rm_dm_total()

        write_excel(dfs, results_folder / "dataset.xlsx")
