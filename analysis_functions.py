import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import random
from sklearn import preprocessing
import sklearn.metrics as sklm
import scipy.stats as ss
import sklearn.model_selection as ms
from joblib import parallel_backend
from rio.machine_learning.transformers.model_transformers.classical_model_transformer import (
    CurveFitModelTransformer,
)


def mse_mod(model, features, label):
    pred = model.predict(features)
    mse = ((pred - label) ** 2).mean()
    return 1 - mse / np.mean(label)


def mare(model, features, label):
    pred = model.predict(features)
    mean_abs_err = np.abs(pred - label).mean()
    return 1 - mean_abs_err / np.mean(label)


def mare_pandas(df, label, predicted_label):

    df["error"] = np.abs(df[label] - df[predicted_label])
    acc_global = 1 - df["error"].mean() / df[label].mean()

    return df, acc_global


def IQR_filtering(data, column_name, IQR_factor, group_feature=None):
    """
    Filters a data frame keeping only the data inside the given IQR. Filters each categorical
    variable in the "group_feature" by the IQR in the column name

    Args:
        data: pandas dataframe contanining the desired data.
        group_feature: column with categorical value, for each value a filter is going
        to be applied
        column_name: name of the feature used to filter
        IQR_factor: factor that multiplies IQR and defines half of the width for the
                    filtering window.

    returns:
        df: filtered dataframe

    """
    if group_feature:
        df = pd.DataFrame()
        for i in data[group_feature].unique():
            #         print(i)
            dfPart = data[data[group_feature] == i]
            Q1 = dfPart[column_name].quantile(0.25)
            Q3 = dfPart[column_name].quantile(0.75)
            IQR = Q3 - Q1
            dfPart = dfPart[
                ~(
                    (dfPart[[column_name]] < (Q1 - IQR_factor * IQR))
                    | (dfPart[[column_name]] > (Q3 + IQR_factor * IQR))
                ).any(axis=1)
            ]
            df = df.append(dfPart)
        return df
    else:
        dfPart = data
        Q1 = dfPart[column_name].quantile(0.25)
        Q3 = dfPart[column_name].quantile(0.75)
        IQR = Q3 - Q1
        dfPart = dfPart[
            ~(
                (dfPart[[column_name]] < (Q1 - IQR_factor * IQR)) | (dfPart[[column_name]] > (Q3 + IQR_factor * IQR))
            ).any(axis=1)
        ]
        return dfPart


def hist_plot(df, cols):
    """
    Plots a histogram of the desired numerical data

    Args:
        vals: data_frame column
        lab: string containing the label
    """
    for col in cols:
        sns.distplot(df[col])
        plt.title("Histogram of " + col)
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.show()


def plot_box(df, col, col_y="hits"):
    """
    Plots a seaborn boxplot

    Args:
        df is a pandas dataframe
        col is a column with the categorical values
        col_y is the column with the quantitative values
    """
    fig, ax = plt.subplots()
    sns.set_style("whitegrid")
    sns.boxplot(col, col_y, data=df, ax=ax)
    plt.xlabel(col)
    plt.ylabel(col_y)
    plt.show()

    return fig, ax


def frequency_table(data, cols):
    """
    Builds a frequency table for the specific columns
    Args:
        df is a pandas dataframe
        cols columns to analyse
    """
    for col in cols:
        print("\n" + "For column " + col)
        print(data[col].value_counts())
        print("There are %s unique values" % data[col].unique().shape[0])


def plot_bars(df, cols):
    """
    Bars for categorical variables
    df: data frame
    cols: columns to plot
    """
    for col in cols:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.gca()
        counts = df[col].value_counts()
        counts.plot.bar(ax=ax, color="blue")
        ax.set_title("Counts" + col)
        ax.set_xlabel(col)
        ax.set_ylabel("Counts")
        plt.show()

    return fig, ax


def plot_scatter(df, cols, col_y, alpha=1.0):
    """
    Scatter plots of the desired columns
    Args:
        df is a pandas dataframe
        cols columns to analyse
        col_y column in the y-axis
    """
    for col in cols:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.gca()
        df.plot.scatter(x=col, y=col_y, ax=ax, alpha=alpha)
        ax.set_title("Scatter plot of " + col_y + " vs. " + col)
        ax.set_xlabel(col)
        ax.set_ylabel(col_y)
        plt.show()


def path_counter(data):
    """
    Return the number of path_ids in the data, assuming that each path is separated by ;
    """

    if isinstance(data, str):
        n_ids = len(data.split(";"))
    elif math.isnan(data):
        n_ids = 0
    else:
        n_ids = 1

    return n_ids


def path_id_splitter(data):
    """
    Splits the path id
    """
    if ";" in data:
        splitted = data.split(";")
        splitted = [float(item) for item in splitted]
    else:
        splitted = [float(data)]
    return splitted


def group_cat(df, col, threshold):
    """
    Groups categorical variables into a group called "other"
    Args:
        df:data_frame
        col:column to group
        threshold: threshold to decide if the variable is grouped or not
    """
    frequencies = pd.DataFrame(df[col].value_counts())
    frequencies = frequencies.reset_index()
    frequencies.columns = [col, "frequency"]
    total = frequencies["frequency"].sum()
    frequencies["perc"] = frequencies["frequency"] / total
    group = frequencies[col].loc[frequencies["perc"] < threshold].values

    return group, frequencies


def cat_aggregation(x, group):
    """
    Assigns the variable "other" to the variables in the group
    """
    if x in group:
        x = "other"
    else:
        x = str(x)
    return x


def encode_string(cat_feature):
    """
    Creates dummy variables out of categorical features and encodes them.
    """
    # First encode the strings to numeric categories
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_feature)
    enc_cat_feature = enc.transform(cat_feature)
    # Now, apply one hot encoding
    ohe = preprocessing.OneHotEncoder()
    encoded = ohe.fit(enc_cat_feature.reshape(-1, 1))
    return encoded.transform(enc_cat_feature.reshape(-1, 1)).toarray()


def print_metrics(y_true, y_predicted):
    # First compute R^2 and the adjusted R^2
    r2 = sklm.r2_score(y_true, y_predicted)

    # Print the usual metrics and the R^2 values
    print("Mean Square Error      = " + str(sklm.mean_squared_error(y_true, y_predicted)))
    print("Root Mean Square Error = " + str(math.sqrt(sklm.mean_squared_error(y_true, y_predicted))))
    print("Mean Absolute Error    = " + str(sklm.mean_absolute_error(y_true, y_predicted)))
    print("Median Absolute Error  = " + str(sklm.median_absolute_error(y_true, y_predicted)))
    print("R^2                    = " + str(r2))


def hist_resids(y_test, y_score):
    # first compute vector of residuals.
    resids = np.subtract(y_test.reshape(-1, 1), y_score.reshape(-1, 1))
    # now make the residual plots
    sns.distplot(resids)
    plt.title("Histogram of residuals")
    plt.xlabel("Residual value")
    plt.ylabel("count")


def resid_qq(y_test, y_score):
    # first compute vector of residuals.
    resids = np.subtract(y_test.reshape(-1, 1), y_score.reshape(-1, 1))
    # now make the residual plots
    ss.probplot(resids.flatten(), plot=plt)
    plt.title("Residuals vs. predicted values")
    plt.xlabel("Predicted values")
    plt.ylabel("Residual")


def resid_plot(y_test, y_score):
    # first compute vector of residuals.
    resids = np.subtract(y_test.reshape(-1, 1), y_score.reshape(-1, 1))
    # now make the residual plots
    sns.regplot(y_score, resids, fit_reg=False)
    plt.title("Residuals vs. predicted values")
    plt.xlabel("Predicted values")
    plt.ylabel("Residual")


def print_format(f, x):
    print("Fold %2d    %4.3f " % (f, x))


def print_cv(scores):
    fold = [x + 1 for x in range(len(scores["test_neg_root_mean_squared_error"]))]
    print("test_neg_root_mean_squared_error")
    [print_format(f, x) for f, x in zip(fold, scores["test_neg_root_mean_squared_error"])]
    print("-" * 40)
    print("Mean       %4.3f" % (np.mean(scores["test_neg_root_mean_squared_error"])))
    print("Std        %4.3f" % (np.std(scores["test_neg_root_mean_squared_error"])))


def train_model(features, label, regression_model, params, n_jobs=-1, scoring=mare):
    """
    Performs a grid search finding the best hyperparameters for the given regression model, based on the mean error (see function below)

    Args:
        features: a numpy array (n_observations, n_features)
        label: 1D array with length n_observations
        regression_model:
        params: parameters for the regression model
        n_jobs: number of runs in parallel
        scoring: function as defined at (3.3.1.3) https://scikit-learn.org/stable/modules/model_evaluation.html#scoring

    Output:
        optimal_model: model after hyperparameter optimisation

    """

    with parallel_backend("threading", n_jobs=n_jobs):
        optimal_model = ms.GridSearchCV(
            estimator=regression_model,
            param_grid=params,
            scoring=scoring,
            cv=3,
            return_train_score=True,
            verbose=2,
        ).fit(features, label)

    return optimal_model


def filtering_dataframe(df, column, center_value, threshold=None, boolean_output=False):
    """
    returns the values of a dataframe around a certain value for a given threshold
    Args:
        df
        column = where the filtering is going to be applied
        center_value = value around which the interval is going to be built
        threshold = Threshold in percentage/100
    Returns:
        df_filtered of booleans if the condition is fullfilled

    """
    if not threshold:
        boolean = df[column] == center_value
    else:
        upper_boundary = center_value * (1 + threshold)
        lower_boundary = center_value * (1 - threshold)
        boolean = (df[column] <= upper_boundary) & (df[column] >= lower_boundary)

    if boolean_output:
        return boolean

    else:
        return df[boolean]


def split_and_train(
    df,
    regressor_model,
    param_grid,
    features,
    label,
    predicted_label=None,
    test_size=0.2,
    random_state=6,
    scoring=mare,
):

    if not predicted_label:
        predicted_label = f"predicted_{label}"

    if test_size != 0:
        df_train, df_test = ms.train_test_split(df, test_size=test_size, random_state=random_state)

        model = train_model(
            df_train[features].values,
            df_train[label].values,
            regressor_model,
            param_grid,
            scoring=scoring,
        )

        df[predicted_label] = model.predict(df[features])

        train_accuracy = model.score(df_train[features], df_train[label])
        test_accuracy = model.score(df_test[features], df_test[label])

    else:
        df_train = df
        model = train_model(
            df_train[features].values,
            df_train[label].values,
            regressor_model,
            param_grid,
            scoring=scoring,
        )

        df[predicted_label] = model.predict(df[features])

        train_accuracy = model.score(df_train[features], df_train[label])
        test_accuracy = 0

    return model, train_accuracy, test_accuracy


def filter_rows_by_values(df, col, values):
    """
    Returns a df that does not contain the values at the given column
    Deletes the rows containing the given values at the desired column
    """
    return df[~df[col].isin(values)]


def model_eval(model, params, label, predicted_label, df):
    """
    Evals the given model (instance of CurveFitModelTransformer)
    for the given df
    """
    features = params["features"]
    data = df[features].to_dict(orient="records")
    results = model.transform(data)
    SHP_pred = [result["pred"] for result in results]
    df[predicted_label] = SHP_pred

    df["error"] = np.abs(df[label] - df[predicted_label])
    acc_global = 1 - df["error"].mean() / df[label].mean()

    return acc_global


def model_fit_and_eval(params, label, predicted_label, df):
    """
    Args:
        params:
    """

    model = CurveFitModelTransformer(**params)
    features = params["features"]
    data = df[features].to_dict(orient="records")
    label_data = df[label].values
    model.fit(data, label_data)
    data = df[features].to_dict(orient="records")
    results = model.transform(data)
    SHP_pred = [result["pred"] for result in results]
    df[predicted_label] = SHP_pred

    df["error"] = np.abs(df[label] - df[predicted_label])
    acc_global = 1 - df["error"].mean() / df[label].mean()

    # Printing the summary
    print(params["function"])
    print(model.model)
    print("The number of features used is %s" % model._n_params)
    print(features)
    print("The global accuracy for the training set is %s" % acc_global)

    return model, acc_global


def add_values_from_df(df_original, df_new, cols, target_column="UTC"):
    """
    adds columns (cols) from df_new into df_original where the target_column(s) are the same.
    Both arrays need to have unique entries in the target_column and must be sortable by the values in such
    column.

    Merging dataframes by using a target column. It adds the rows from previos
    Adds the rows from df_new to df_original when the column is the same
    df_original: dataFrame where the rows are going to be filled with data from the new dataframe
    df_new: dataFrame with new values
    cols: cols to be replaced in df_original
    target_column: common column between the two dataframes that is going to be used to merge.


    """
    df_original.sort_values(target_column, inplace=True)
    df_new.sort_values(target_column, inplace=True)

    df_original.loc[df_original[target_column].isin(df_new[target_column]), cols] = df_new.loc[
        df_new[target_column].isin(df_original[target_column]), cols
    ].values

    return df_original


def model_score_per_category(df, category_column, model, features, label):
    """
    Print the model score for each element in a category given by a defined column
    """
    categories = df[category_column].unique()

    for category in categories:
        df_category = df[df[category_column] == category]
        number_in_category = df_category.shape[0]
        if number_in_category > 0:
            score = model.score(df_category[features], df_category[label])
            print(category, score, number_in_category)


def train_test_split_by_category(df, category_label, object_id, train_percentage):
    """
    object_id: identifier for each object within the category. The splitting takes into account the object_id, such that one object_id (i.e IMO) only belong to one group
    category_label: Vessel type for example
    """
    random.seed(2)
    df_train = []
    df_test = []
    for category in df[category_label].unique():
        if category == "Wood Chips Carrier":
            continue
        print(category)
        df_category = df[df[category_label] == category]
        object_id_list = sorted(df_category[object_id].unique())
        n_objects_in_category = len(object_id_list)
        n_objects_train = math.floor((train_percentage / 100) * n_objects_in_category)
        print(f"{n_objects_train}/{n_objects_in_category}")
        train_object_ids = random.sample(object_id_list, k=n_objects_train)
        test_object_ids = [id for id in object_id_list if id not in train_object_ids]

        for object_id_train in train_object_ids:
            df_train.append(df_category[df_category[object_id] == object_id_train])

        for object_id_test in test_object_ids:
            df_test.append(df_category[df_category[object_id] == object_id_test])

        print(train_object_ids)
        print(test_object_ids)

    df_train = pd.concat(df_train)
    df_test = pd.concat(df_test)

    return df_train, df_test
