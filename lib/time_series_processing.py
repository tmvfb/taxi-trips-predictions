import itertools

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL
from matplotlib import cm
from sklearn.model_selection import TimeSeriesSplit, cross_validate, train_test_split
from tqdm import tqdm


class TSDataManager:
    """
    A class to manage pandas time series data.
    Pass only index and features you want to create lag/rolling mean for.

    Methods
    -------
    __init__(
        self,
        df: pd.core.frame.DataFrame,
        decomposed_df: statsmodels.tsa.seasonal.DecomposeResult = None,
        label: str = "count"
    ):
        Creates class instance.

    make_features(max_lag: int, rolling_mean_size: int, add_decomposed: bool = False):
        Performs feature engineering for a given DataFrame.
        Adds lag columns, rolling mean column and STL decomposed trend + seasonal data.
        Returns passed feature engineering parameters dictionary and modified df.

    split():
        Train test split of the passed data. Returns dict of split samples.

    grid_search(model, params: dict):
        Finds the best values for number of lag columns, rolling mean size and
        presence of the decomposed data.
        Returns a dictionary with best parameters and best CV score.

    plot(plot_non_decomposed: bool = True):
        Builds a plot for the performed grid search.
    """

    def __init__(
        self,
        df: pd.core.frame.DataFrame,
        label: str = "count"
    ):
        self.df = df.copy()
        self.df_cleared = df.copy()
        self.label = label

    def make_features(
        self, max_lag: int, rolling_mean_size: int, add_decomposed: bool = False
    ) -> tuple:
        self.df = self.df_cleared.copy()  # use initial df
        params = {
            "max_lag": max_lag,
            "rolling_mean_size": rolling_mean_size,
            "add_decomposed": add_decomposed,
        }

        for column in self.df.columns:
            for lag in range(1, max_lag + 1):
                self.df[f"{column}_lag_{lag}"] = self.df[column].shift(lag)

            self.df[f"{column}_rolling_mean"] = (
                self.df[column].shift().rolling(rolling_mean_size).mean()
            )

            if add_decomposed:  # adding STL features
                df_decomposed = STL(self.df[column], robust=True).fit()  # slow
                self.df[f"{column}_trend"] = df_decomposed.trend.values
                self.df[f"{column}_seasonal"] = df_decomposed.seasonal.values
                self.df[f"{column}_trend"] = self.df[f"{column}_trend"].shift()
                self.df[f"{column}_seasonal"] = self.df[f"{column}_seasonal"].shift()

            if column != self.label:
                self.df.drop(column, axis=1, inplace=True)

        self.df["hour"] = self.df.index.hour  # the task is to predict for next hour
        self.df["month"] = self.df.index.month
        self.df["dayofweek"] = self.df.index.dayofweek

        # self.df["day"] = self.df.index.day
        # self.df["count_lag_YEAR"] = self.df[self.label].shift(8760)
        # self.df["count_lag_YEARandHOUR"] = self.df[self.label].shift(8761)
        # self.df["count_lag_YEARandWEEK"] = self.df[self.label].shift(8928)

        # self.df.dropna(inplace=True)

        return params, self.df

    def split(self) -> dict:
        train, test = train_test_split(self.df.dropna(), shuffle=False, test_size=0.1)
        X_train = train.drop(self.label, axis=1)
        y_train = train[self.label]
        X_test = test.drop(self.label, axis=1)
        y_test = test[self.label]

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    def grid_search(self, model, params: dict) -> dict:
        # sort to work correctly with itertools:
        params_order = ["max_lag", "rolling_mean_size", "add_decomposed"]
        params = {key: params[key] for key in params_order if key in params}

        # collect all parameter combinations in a list
        parameter_combinations = []
        for key in params:
            parameter_combinations.append(params[key])
        parameter_combinations = list(itertools.product(*parameter_combinations))

        # params to return + params for plot
        best_params = {}
        self.plotting_params = {
            "X": [],
            "Y": [],
            "Z": [],
            "decomposed_features_added": [],
        }
        best_score = 10000

        # best params for feature engineering
        for combination in tqdm(parameter_combinations):
            current_params, _ = self.make_features(*combination)  # generate features
            samples = self.split()  # train_test_split
            tscv = TimeSeriesSplit(n_splits=8)

            cv = cross_validate(
                X=samples["X_train"],
                y=samples["y_train"],
                estimator=model,
                scoring="neg_root_mean_squared_error",
                cv=tscv,
                n_jobs=-1,
                return_estimator=True,
            )

            # get plotting params
            self.plotting_params["X"].append(current_params["max_lag"])
            self.plotting_params["Y"].append(current_params["rolling_mean_size"])
            self.plotting_params["Z"].append(-cv["test_score"].mean())
            self.plotting_params["decomposed_features_added"].append(
                current_params["add_decomposed"]
            )

            # remember best params
            if -cv["test_score"].mean() < best_score:
                best_score = -cv["test_score"].mean()
                best_params = current_params

        # return fitted model
        current_params, _ = self.make_features(**best_params)
        samples = self.split()
        model.fit(samples["X_train"], samples["y_train"])

        return {
            "best_score_RMSE": best_score,
            "best_params": best_params,
            "best_model": model,
        }

    def plot(self, plot_non_decomposed: bool = True):
        plot_data = pd.DataFrame(self.plotting_params)
        plot_data_true = plot_data[plot_data["decomposed_features_added"] == True]  # noqa
        plot_data_false = plot_data[plot_data["decomposed_features_added"] == False]  # noqa

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d")

        ax.set_title("RMSE train vs lag/rolling mean")
        ax.set_xlabel("number of lag columns", fontweight="bold")
        ax.set_ylabel("rolling mean size", fontweight="bold")
        ax.set_zlabel("RMSE", fontweight="bold", labelpad=30)
        ax.ticklabel_format(axis="z", style="plain")
        ax.tick_params(axis="z", which="major", pad=12)

        ax.plot_trisurf(
            plot_data_true["X"],
            plot_data_true["Y"],
            plot_data_true["Z"],
            cmap=cm.RdYlGn,
            linewidth=2,
            antialiased=True,
        )
        legend_labels = [
            mpatches.Patch(color="green", label="data with decomposed time features")
        ]

        # plot second figure:
        if plot_non_decomposed:
            ax.plot_trisurf(
                plot_data_false["X"],
                plot_data_false["Y"],
                plot_data_false["Z"],
                cmap=cm.Purples,
                linewidth=2,
                antialiased=True,
            )
            legend_labels.append(
                mpatches.Patch(
                    color="purple", label="data without decomposed time features"
                )
            )

        ax.legend(handles=legend_labels)
