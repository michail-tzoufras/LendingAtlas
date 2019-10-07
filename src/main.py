# Developed by: Michail Tzoufras
# Date updated: 10/6/2019

import os
import csv
import argparse
import sys

import pandas as pd
import numpy as np
import copy

# ------------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ------------------------------------------------------------------

import visualization as Vis
import data_processing as Process
import utilities as Utils

# ------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        action="store",
        default=os.getcwd() + "/../data/processed/processed_kiva_data.csv",
        help="dataset path",
    )
    parser.add_argument(
        "--solver",
        action="store",
        default="All",
        help="Select solver from: (1) 'All' (2) 'Random Forest' (3) 'Embeddings' (4) 'Logistic Regression' ",
    )
    parser.add_argument(
        "--epochs",
        metavar="N",
        type=Utils.check_positive,
        nargs="+",
        default=[50, 50],
        help="The number of epochs to train each network for",
    )
    parser.add_argument(
        "--batch_size",
        type=Utils.check_positive,
        nargs="+",
        default=[500, 500],
        help="The batch size when training each network",
    )
    parser.add_argument(
        "--shallow_net",
        metavar="N",
        type=Utils.check_positive,
        nargs="+",
        default=[32, 8],
        help="number of nodes for each layer of the network used to train the embeddings",
    )
    parser.add_argument(
        "--deep_net",
        metavar="N",
        type=Utils.check_positive,
        nargs="+",
        default=[],
        help="number of nodes for each layer of the network that uses the trained embeddings",
    )
    parser.add_argument(
        "--sample",
        action="store",
        default="undersample",
        help="For imbalanced classes: (1) 'undersample' (2) 'oversample' (3) 'None' ",
    )
    parser.add_argument(
        "--explore_data",
        type=Utils.str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Generate data exploration plots",
    )

    args = parser.parse_args()

    # Utils.split(open(args.data, 'r'), datapath, dataname)

    df_raw = pd.read_csv(args.data)

    df_raw["Country Currency"] = "Empty"
    for i in range(len(df_raw)):
        df_raw.at[i, "Country Currency"] = (
            df_raw["Country"].iloc[i] + " " + df_raw["Currency"].iloc[i]
        )

    useful_columns = [
        "Loan Amount",
        "Loan Term",
        "Town",
        "Country",
        "Sector",
        "Activity",
        "Partner ID",
        "Country Currency",
        "Status",
    ]
    valid_status = ["paid", "defaulted"]

    df_clean_by_status = (
        df_raw[useful_columns][df_raw["Status"].isin(valid_status)]
    ).copy()

    # A list of countries to exclude from the dataset because they either have 100% default rate (i.e. Gaza and Belize) or 0% (see below)
    countries_to_exclude = [
        "Nigeria",
        "Vietnam",
        "Paraguay",
        "Ukraine",
        "Mali",
        "Congo",
        "Albania",
        "Sri Lanka",
        "Zambia",
        "Timor-Leste",
        "Cote D'Ivoire",
        "Bosnia and Herzegovina",
        "South Africa",
        "Nepal",
        "Moldova",
        "Yemen",
        "Gaza",
        "Belize",
    ]
    df_clean = (
        df_clean_by_status[useful_columns][
            ~df_clean_by_status.Country.isin(countries_to_exclude)
        ]
    ).copy()

    # A new feature "Funded Time" gives the exact time when the loan was funded.
    df_clean["Funded Time"] = (
        (df_raw["Funded Date.year"] + 0.0833 * df_raw["Funded Date.month"])[
            df_raw.Status.isin(valid_status)
        ]
    ).copy()

    # Make the directory for the output figures if there isn't one
    output_figs_path = os.getcwd() + "/output_figs/"
    if not os.path.isdir(output_figs_path):
        os.mkdir(output_figs_path)

    # Identify the categorical and ordinal features
    categorical_columns = [
        "Country",
        "Sector",
        "Activity",
        "Partner ID",
        "Country Currency",
        "Town",
    ]
    ordinal_columns = ["Loan Amount", "Loan Term", "Funded Time"]
    # The remaining category is "Status" and this will be the supervised
    # task to train on

    # Initialize the visualization class
    visualize = Vis.Make_Visualizations(output_figs_path)

    # Visualize some data exploration
    if args.explore_data:
        visualize.data_exploration(df_clean)
        visualize.country_vs_status(df_clean)

    # Split data set into training and test sets
    df_train_raw, df_test = train_test_split(df_clean, test_size=0.3, random_state=3)

    # Undersample the majority (default) or ...
    if args.sample == "undersample":
        df_train = Process.undersample_majority(df_train_raw, 1)
    # ... oversample the minority or ...
    elif args.sample == "oversample":
        df_train = Process.oversample_minority(df_train_raw, 1)
    # ... work with imbalance as is
    else:
        df_train = df_train_raw

    # Concatenate the train and test dataframes
    # to make sure the arrays have the same number
    # of features when they are split.
    train_length = len(df_train)
    df_concatenated = pd.concat([df_train, df_test])

    # Convert dataframe to numpy array and labels
    X_concatenated = Process.dataframe_to_numpy(
        df_concatenated, categorical_columns, ordinal_columns
    )
    y_concatenated = np.array(
        (
            pd.get_dummies(df_concatenated["Status"], columns=["Status"])["defaulted"]
        ).tolist()
    )

    # Split the train from the test arrays
    X_train, X_test = X_concatenated[:train_length, :], X_concatenated[train_length:, :]
    y_train, y_test = y_concatenated[:train_length], y_concatenated[train_length:]

    y_pred, y_prob, model_titles = [], [], []
    if (
        (args.solver == "Logistic Regression")
        or (args.solver == "Random Forest")
        or (args.solver == "All")
    ):

        if (args.solver == "Logistic Regression") or (args.solver == "All"):
            # Import module for fitting
            logmodel = LogisticRegression(
                solver="lbfgs",  # class_weight= {0:.1, 1:.9},
                penalty="l2",
                C=10.0,
                max_iter=500,
            )

            # Fit the model using the training data
            logmodel.fit(X_train, y_train)

            y_pred.append(logmodel.predict(X_test))
            y_prob.append(logmodel.predict_proba(X_test)[:, 1])
            model_titles.append("Logistic Regression")

        if (args.solver == "Random Forest") or (args.solver == "All"):
            # Import module for fitting
            rfmodel = RandomForestClassifier(
                n_estimators=80, max_depth=50  # class_weight= {0:.1, 1:.9},
            )

            # Fit the model using the training data
            rfmodel.fit(X_train, y_train)

            y_pred.append(rfmodel.predict(X_test))
            y_prob.append(rfmodel.predict_proba(X_test)[:, 1])
            model_titles.append("Random Forest")

    if (args.solver == "Embeddings") or (args.solver == "All"):

        import embeddings_DL as Emb

        # Find the vocabulary sizes for the categorical features
        vocabulary_sizes = [df_concatenated[c].nunique() for c in categorical_columns]

        # Maximum sentence length
        max_length = 1

        # Encode categorical data
        categorical_encoder = Emb.One_Hot_Encoder(df_concatenated, categorical_columns)
        categorical_data = categorical_encoder.encode(
            df_concatenated, categorical_columns, max_length
        )

        # normalize the ordinal features
        ordinal_data = [
            (Process.normalize_column(df_concatenated[c])).reshape(-1, 1)
            for c in ordinal_columns
        ]

        # merge input data and split to train and test
        input_data = categorical_data + ordinal_data
        input_data_train = [data[:train_length, :] for data in input_data]
        input_data_test = [data[train_length:, :] for data in input_data]

        # Shallow network to obtain embeddings
        emb_model = Emb.embeddings_models(
            vocabulary_sizes,
            max_length,
            categorical_columns,
            len(ordinal_columns),
            visualize,
        )
        train_embeddings_model = emb_model(
            input_data_train,
            y_train,
            input_data_test,
            y_test,
            args.epochs[0],
            args.batch_size[0],
            args.shallow_net,
        )

        y_pred.append(train_embeddings_model.predict(input_data_test))
        y_prob.append(train_embeddings_model.predict_prob(input_data_test))
        model_titles.append("Train Embeddings")

        # Make an output directory for embeddings if there isn't one
        output_embeddings = os.getcwd() + "/output_embeddings/"
        if not os.path.isdir(output_embeddings):
            os.mkdir(output_embeddings)

        # Extract and save embeddings in a csv file
        embs = Emb.save_embeddings(
            output_embeddings,
            train_embeddings_model,
            categorical_columns,
            categorical_encoder,
        )

        # Make the visualization class for the embeddings
        embeddings_atlas = Vis.Visualize_Embeddings(
            output_embeddings, output_figs_path, df_clean, categorical_columns
        )
        for c in categorical_columns:
            if c == "Country":
                embeddings_atlas.display(
                    c, ["United States", "Philippines", "Mozambique", "Bulgaria"]
                )
            else:
                embeddings_atlas.display(c)

        # For each country in the list  below highlight its towns in
        # the TSNA of the town embedding
        highlighted_countries = ["Guatemala", "Uganda", "Afghanistan"]
        for highlighted_country in highlighted_countries:
            highlighted_towns = []
            for i in (
                df_clean[df_clean["Country"] == highlighted_country]["Town"]
            ).unique():
                highlighted_towns.append(i)
            if "Rare_Town" in highlighted_towns:
                highlighted_towns.remove("Rare_Town")

            embeddings_atlas.display_towns(c, highlighted_country, highlighted_towns)

        # Plot the cosine similarity (highest and lowest values)
        # for this feature/value
        embeddings_atlas.plot_embeddings_similarity("Country Currency", "Lebanon LBP")

        # Use embeddings in a deeper network
        if len(args.deep_net) > 0:
            embs_list = list(embs.values())
            deploy_embeddings_model = emb_model(
                input_data_train,
                y_train,
                input_data_test,
                y_test,
                args.epochs[1],
                args.batch_size[1],
                args.deep_net,
                embs_list,
            )

            y_pred.append(deploy_embeddings_model.predict(input_data_test))
            y_prob.append(deploy_embeddings_model.predict_prob(input_data_test))
            model_titles.append("Pretrained Embeddings")

    visualize.report_model_performance(y_test, y_pred, y_prob, model_titles)
