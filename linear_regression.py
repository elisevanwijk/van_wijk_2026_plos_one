import pandas as pd
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant

from common import results_folder, write_excel
from data_loader import DataLoader


def regression_report(model):
    summary = str(model.summary())
    test_string = "z_score_Invulvraag = z_score_Meerkeuzevraag"
    f_test = model.f_test(test_string)
    t_test = model.t_test(test_string)
    wald_test = model.wald_test(test_string, scalar=True)

    summary += "-----------------------------\n"
    summary += f"Wald\t{test_string} \tp={wald_test.pvalue:.4f}\n"
    summary += f"f_test\t{test_string}\tp={f_test.pvalue:.4f}\n"
    summary += f"t_test\t{test_string}\tp={t_test.pvalue:.4f}\n"

    return summary


class LinearRegression:
    def __init__(self):
        self.data_loader = DataLoader()

    rm_dependent_variables = ["gpa_last_attempt_year_1", "gpa_first_attempt_year_1"]
    rm_dm_dependent_variables = ["gpa_last_attempt_total", "gpa_first_attempt_total"]

    independent_variables = ["z_score_Invulvraag", "z_score_Meerkeuzevraag"]

    def _run_regressions(self, sheets, dependent_variables, independent_variables):
        return {
            sheet_name: self._run_regression(
                data, dependent_variables, independent_variables
            )
            for sheet_name, data in sheets
        }

    def _run_regression(self, data, dependent_variables, independent_variables):
        results = {}
        for dependent_variable in dependent_variables:
            x = add_constant(data[independent_variables])
            model = sm.OLS(
                data[dependent_variable],
                x,
            ).fit()
            results[dependent_variable] = model

        return results

    def classify_per_cohort_rm(self):
        return self._run_regressions(
            self.data_loader.rm_per_cohort(),
            self.rm_dependent_variables,
            self.independent_variables,
        )

    def report_classify_per_cohort_rm(self):
        for sheet_name, models in self.classify_per_cohort_rm().items():
            print(sheet_name)
            for _, model in models.items():
                print(regression_report(model))

    def classify_per_cohort_rm_dm(self):
        return self._run_regressions(
            self.data_loader.rm_dm_per_cohort(),
            self.rm_dm_dependent_variables,
            self.independent_variables,
        )

    def report_classify_per_cohort_rm_dm(self):
        for sheet_name, models in self.classify_per_cohort_rm_dm().items():
            print(sheet_name)
            for _, model in models.items():
                print(regression_report(model))

    def classify_rm_total(self):
        return self._run_regression(
            self.data_loader.rm_total(),
            self.rm_dependent_variables,
            self.independent_variables,
        )

    def report_classify_rm_total(self):
        for _, model in self.classify_rm_total().items():
            print(regression_report(model))

    def classify_rm_dm_total(self):
        return self._run_regression(
            self.data_loader.rm_dm_total(),
            self.rm_dm_dependent_variables,
            self.independent_variables,
        )

    def report_classify_rm_dm_total(self):
        for _, model in self.classify_rm_dm_total().items():
            print(regression_report(model))

    def generate_rm_results(self):
        model_runs = self.classify_per_cohort_rm()
        model_runs["RM Total"] = self.classify_rm_total()
        return self.generate_results(model_runs, "rm")

    def generate_rm_dm_results(self):
        model_runs = self.classify_per_cohort_rm_dm()
        model_runs["RM DA Total"] = self.classify_rm_dm_total()
        return self.generate_results(model_runs, "rm_da")

    def generate_results(self, model_runs, label):
        coefs = pd.concat(
            [self.get_coefs(models, sheet) for sheet, models in model_runs.items()]
        )
        model_results = pd.concat(
            [
                self.get_model_results(models, sheet)
                for sheet, models in model_runs.items()
            ]
        )
        logs = self.generate_log(model_runs)

        dfs = {"Coefficients": coefs, "Model results": model_results}

        write_excel(dfs, results_folder / f"linear_regression_{label}.xlsx", False)
        for log_name, result in logs.items():
            pth = results_folder / f"{label}_ols_logs" / f"{log_name}.txt"
            pth.parent.mkdir(exist_ok=True, parents=True)
            pth.write_text(result)
        return coefs

    def generate_log(self, model_runs):
        logs = {}
        for sheet, models in model_runs.items():
            results = []
            for dep, model in models.items():
                result = f"{dep}\n"
                result += regression_report(model)
                results.append(result)

            logs[sheet] = "\n\n----------------\n\n".join(results)
        return logs

    def get_coefs(self, models, label):
        return pd.concat(
            [self.get_coef(model, label) for model in models.values()],
            axis=1,
            keys=models.keys(),
        )

    def get_model_results(self, models, label):
        return pd.concat(
            [self.get_model_result(model, label) for model in models.values()],
            axis=1,
            keys=models.keys(),
        )

    def get_model_result(self, model, label):
        model_params = pd.Series(
            [
                model.rsquared,
                model.rsquared_adj,
                model.df_model,
                model.df_resid,
                model.fvalue,
                model.f_pvalue,
            ],
            index=[
                "R Squared",
                "R Squared adj",
                "DF model",
                "Df residuals",
                "F statistic",
                "F p-value",
            ],
        ).to_frame()
        model_params.columns = [label]

        return model_params.T

    def get_coef(self, model, label):
        params = model.params
        features = list(params.index.drop("const"))
        conf = model.conf_int()

        conf = pd.concat([model.pvalues, model.conf_int(), model.tvalues], axis=1)
        conf.columns = ["p-values", "5%", "95%", "t"]
        conf["coef"] = params
        conf["se"] = model.bse
        features = {f: "VSAQ" if "Invul" in f else "MCQ" for f in features}
        conf = conf.rename(features)
        conf = conf.unstack().to_frame()
        conf.columns = [label]
        order = ["coef", "se", "5%", "95%", "t", "p-values"]
        conf = conf.swaplevel().sort_index(level=0).reindex(order, level=1)

        return conf.T
