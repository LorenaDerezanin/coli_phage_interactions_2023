import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import linkage
import scipy
import seaborn as sns
import json
import pickle

import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve, average_precision_score, f1_score

np.random.seed(0)

save_dir = "D:\\These\\30_dev\\302_HR_predictions\\+++\\stratified_all_phages\\adsorption_factors+core_genome_1e-4_quat"
if not os.path.isdir(save_dir + "\\results"):
    os.mkdir(save_dir + "\\results")
if not os.path.isdir(save_dir + "\\results\\models"):
    os.mkdir(save_dir + "\\" + "results\\models")
if not os.path.isdir(save_dir + "\\results\\feature_importances"):
    os.mkdir(save_dir + "\\" + "results\\feature_importances")
if not os.path.isdir(save_dir + "\\results\\logs"):
    os.mkdir(save_dir + "\\" + "results\\logs")
if not os.path.isdir(save_dir + "\\results\\performances"):
    os.mkdir(save_dir + "\\" + "results\\performances")
if not os.path.isdir(save_dir + "\\results\\predictions"):
    os.mkdir(save_dir + "\\" + "results\\predictions")

# Load data
interaction_matrix = pd.read_csv("D:\\These\\20_data\\212_interaction_matrices\\ADSORPTION_MATRIX_370+host_vs_96_less_stringent.csv", sep=";").set_index("bacteria")
interaction_matrix = interaction_matrix.replace({"N": 0, "P": 1, "F": np.nan, "U": 1})

# interaction_matrix = interaction_matrix.loc[interaction_matrix.count(axis=1)[interaction_matrix.count(axis=1) > 70].index].fillna(0)

phage_feat_names = ["Morphotype", "Genus", "Phage_host"]
print(f"Phage features : {phage_feat_names}")

phage_features = pd.read_csv("D:\These\\20_data\\201_genomic_data\\097_phage\\Phage_features_with_host.csv", sep=";").set_index("phage").loc[interaction_matrix.columns, phage_feat_names]
bact_features = pd.read_csv("D:\These\\20_data\\201_genomic_data\\370_and_host\\370+host_features_with_ABC_capsules_2.csv", sep=";").set_index("bacteria")
cv_clusters = pd.read_csv("D:\\These\\30_dev\\302_HR_predictions\\370+host_cross_validation_groups_1e-4.csv", sep=";").set_index("bacteria")

bact_embeddings = pd.read_csv("D:\\These\\30_dev\\302_HR_predictions\\core_genome\\UMAP_dim_reduction_from_phylogeny\\data\\coli_umap_8_dims.tsv", sep="\t").set_index("bacteria")

bact_features = pd.merge(bact_features, bact_embeddings, left_index=True, right_index=True)

bact_feat_names = "(UMAP|O-type|LPS|ST_Warwick|Klebs|ABC_serotype)"
bact_features = bact_features.filter(regex=bact_feat_names, axis=1)

for p in phage_features.index:
    print(f"Processing phage {p}...")

    # Filter phages according to phylogeny
    phage_feat = phage_features.loc[[p]]
    interaction_mat = interaction_matrix[[p]]

    phage_feat = phage_feat.drop(["Morphotype", "Genus"], axis=1)

    # wide to long
    interaction_matrix_long = interaction_mat.unstack().reset_index().rename({"level_0": "phage", 0: "y"}, axis=1).sort_values(["bacteria", "phage"])  # force row order

    # Add the cross-validation index of each observation for Leave-one-strain-out CV
    interaction_matrix_long = pd.merge(interaction_matrix_long, cv_clusters, left_on=["bacteria"], right_index=True).set_index("group")
    groups = interaction_matrix_long.index

    # Concat features and target
    interaction_with_features = pd.merge(interaction_matrix_long, bact_features, left_on=["bacteria"], right_index=True)

    # Add phage host features to predictors
    phage_host_features = pd.merge(phage_feat, bact_features.filter(regex="(ST_Warwick|O-type|H-type)", axis=1), left_on="Phage_host", right_index=True).rename({"Clermont_Phylo": "Clermont_host", "LPS_type": "LPS_host", "O-type": "O_host", "H-type": "H_host", "ST_Warwick": "ST_host"}, axis=1)

    if not p.startswith("LF110"):  # do not have the data for LF110 host strain
        interaction_with_features = pd.merge(interaction_with_features, phage_host_features.drop(["Phage_host"], axis=1), left_on="phage", right_index=True)

    # Recode O-type : only keep main categories to avoid having too many levels
    if "O-type" in bact_features.columns:
        otypes_to_recode = bact_features.groupby("O-type").filter(lambda x: x.shape[0] < 3)["O-type"].unique()  # less than 5 observations for the O-type value
        interaction_with_features.loc[interaction_with_features["O-type"].isin(otypes_to_recode), "O-type"] = "Other"
        if not p.startswith("LF110"):
            interaction_with_features["same_O_as_host"] = interaction_with_features["O-type"] == interaction_with_features["O_host"]
            interaction_with_features = interaction_with_features.drop("O_host", axis=1)

    # Recode ST : only keep main categories to avoid having too many levels
    if "ST_Warwick" in bact_features.columns:
        st_to_recode = bact_features.groupby("ST_Warwick").filter(lambda x: x.shape[0] < 3)["ST_Warwick"].unique()  # less than 5 observations for the O-type value
        interaction_with_features.loc[interaction_with_features["ST_Warwick"].isin(st_to_recode), "ST_Warwick"] = "Other"
        if not p.startswith("LF110"):
            interaction_with_features["same_ST_as_host"] = interaction_with_features["ST_Warwick"] == interaction_with_features["ST_host"]

    if "ABC_serotype" in bact_features.columns:
        if not p.startswith("LF110"):
            interaction_with_features["same_ABC_as_host"] = interaction_with_features["ABC_serotype"] == interaction_with_features["ABC_serotype"]

    if "same_O_as_host" in interaction_with_features.columns and "same_ST_as_host" in interaction_with_features.columns and not p.startswith("LF110"):
        interaction_with_features["same_O_and_ST_as_host"] = interaction_with_features["same_O_as_host"] * interaction_with_features["same_ST_as_host"]

    # Drop missing observations
    na_observations = interaction_with_features.loc[interaction_with_features["y"].isna()].index
    interaction_with_features = interaction_with_features.drop(na_observations, axis=0)

    # Dummy encoding of categorical variables and standardization for numerical variables
    X, y, bact_phage_names = interaction_with_features.drop(["bacteria", "phage", "y"], axis=1), interaction_with_features["y"], interaction_with_features[["bacteria", "phage"]]

    num, factors = [], []
    for col_dtype, col in zip(X.dtypes, X.dtypes.index):
        if col_dtype == "float64":
            num.append(col)
        else:
            factors.append(col)
    X_oh = pd.concat([(X[num] - X[num].mean(axis=0)) / X[num].std(axis=0), pd.get_dummies(X[factors], sparse=False)], axis=1)

    # Perform cross-validation
    import warnings
    from sklearn.exceptions import UndefinedMetricWarning
    warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)  # shutdown sklearn warning regarding ill-defined precision

    def get_alias(model):
        aliases = {LogisticRegression: "LogReg", RandomForestClassifier: "RF", DummyClassifier: "Dummy", MLPClassifier: "MLP",
                BernoulliNB: "NaiveBayes", DecisionTreeClassifier: "DecTree"}
        name = aliases[type(model)]
        if type(model) == LogisticRegression:
            name += "_" + model.penalty
        elif type(model) == RandomForestClassifier:
            name += "_" + str(model.n_estimators) + "_" + str(model.max_depth)
        elif type(model) == DummyClassifier:
            name += "_" + model.strategy
        elif type(model) == MLPClassifier:
            hidden_layer_sizes = list(str(x) for x in model.get_params()["hidden_layer_sizes"])
            name += "_" + "-".join(hidden_layer_sizes) + "_lr=" + str(model.get_params()["learning_rate_init"])
        if hasattr(model, "class_weight") and model.class_weight is not None:
            name += "_weight=" + str(model.class_weight[1])
        return name

    from sklearn.exceptions import NotFittedError

    def perform_group_cross_validation(X, y, models, models_params, n_splits=10, groups=None, index_names=None, do_scale=False):
        group_kfold = GroupKFold(n_splits=n_splits)
        umap_dim = X.shape[1] // 2

        if groups is None:
            raise ValueError("Should provide grouping variable in order to perform Group-K-fold CV.")
        
        # Train feature scaler on the whole dataset (if required)
        if do_scale:
            std_scaler = MinMaxScaler()
            std_scaler.fit(X)

        performance, predictions, logs = [], [], []
        model_list = {}
        for i, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups)):  # K-fold cross-validation
            X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
            chosen_groups = groups[test_idx].unique().values

            # check that train set observations and validation set observations are disjoint
            assert(set(X_train.index).intersection(set(X_test.index)) == set())
            
            for model_type, param in zip(models, models_params):
                
                model = model_type(**param)
                alias = get_alias(model)

                # Fit model (train set)
                model.fit(X_train, y_train)

                # Model evaluation (train and test set)
                for (ds, ds_name) in zip([[X_train, y_train], [X_test, y_test]], ["train", "test"]):
                    xset, yset = ds

                    # Feature scaling (if required)
                    if do_scale:
                        xset = pd.DataFrame(std_scaler.transform(xset), columns=X.columns)

                    # Predictions
                    y_pred, y_pred_proba = model.predict(xset), model.predict_proba(xset)

                    # Metrics
                    if np.unique(yset).shape[0] > 1:  # Cannot compute metrics if only one class is predicted
                        tn, fp, fn, tp = confusion_matrix(yset, y_pred).ravel()
                        precision, recall, f1 = precision_score(yset, y_pred), recall_score(yset, y_pred), f1_score(yset, y_pred)
                        average_prec = average_precision_score(yset, y_pred_proba[:, 1])
                        roc_auc = roc_auc_score(yset, y_pred_proba[:, 1])

                        performance.append({"model": alias, "fold": i, "dataset": ds_name, "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc, "avg_precision": average_prec,
                                        "tp": tp, "fp": fp, "tn": tn, "fn": fn,})
                    else:
                        performance.append({"model": np.nan, "fold": np.nan, "dataset": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "roc_auc": np.nan, "avg_precision": np.nan,
                                        "tp": np.nan, "fp": np.nan, "tn": np.nan, "fn": np.nan,})

                    # Collect predictions (test set only)
                    if ds_name == "test": #and not alias.startswith("Dummy"):
                        preds = index_names.iloc[test_idx].copy()
                    else:
                        preds = index_names.iloc[train_idx].copy()
                    preds["y_pred"] = model.predict(xset)
                    preds["y_pred_proba"] = model.predict_proba(xset)[:, 1]
                    preds["fold"] = i
                    preds["model"] = alias
                    preds["dataset"] = ds_name
                    predictions.append(preds)  # add bacteria-phage name as index instead of integer (avoid ambiguity)

                model_list[f"{p}_{alias}_fold={i}"] = model
                del model
            logs.append({"fold": i, "train_size": train_idx.shape[0], "test_size": test_idx.shape[0], "train_idx": train_idx, "test_idx": test_idx, "cv_groups": chosen_groups})

        logs = pd.DataFrame(logs)
        performance = pd.DataFrame(performance)
        all_cv_predictions = pd.concat([pred for pred in predictions])[["fold", "model", "dataset", "bacteria", "phage", "y_pred_proba", "y_pred"]]

        return logs, performance, all_cv_predictions, model_list

    n_splits = 10
    redo_predictions = True
    if redo_predictions:  # avoid overwriting predictions by mistake

        # Make predictions
        models_to_test =  [
                            RandomForestClassifier,
                            RandomForestClassifier,
                            LogisticRegression,
                            LogisticRegression,
                            DummyClassifier
                        ]
        
        # choose class weight
        perc_pos_class = y.sum() / y.shape[0]
        if 0.60 <= perc_pos_class:
            cw = {0:1, 1:0.8}
        elif 0.4 <= perc_pos_class < 0.6:
            cw = {0:1, 1: 1}
        elif 0.3 <= perc_pos_class < 0.4:
            cw = {0:1, 1: 1.5}
        elif 0.2 <= perc_pos_class < 0.3:
            cw = {0:1, 1: 2}
        else:
            cw = {0:1, 1: 3}

        # cw = "balanced"

        params = [
                    {"max_depth": 3, "n_estimators": 250, "class_weight": cw},
                    {"max_depth": 6, "n_estimators": 250, "class_weight": cw},
                    {"class_weight": cw, "max_iter": 10000},
                    {"class_weight": cw, "penalty": "l1", "solver": "saga", "max_iter": 10000},
                    {"strategy":"stratified"}
                ]
        logs, performance, cv_predictions, trained_models = perform_group_cross_validation(X_oh, y, n_splits=n_splits, groups=interaction_with_features.index, 
                                                                                        index_names=bact_phage_names, 
                                                                                        models=models_to_test, models_params=params, do_scale=False)
        
        performance["phage"] = p
        cv_predictions["phage"] = p

        performance = performance.set_index("phage")
        cv_predictions = cv_predictions.set_index("phage")
        
        cv_predictions = pd.merge(cv_predictions, interaction_with_features[["bacteria", "phage", "y"]], on=["bacteria", "phage"])  # add real interaction values

        overwrite_files = True  # overwrite log and performance files
        if overwrite_files:
            logs.to_csv(f"{save_dir}\\results\\logs\\logs_{p}_Group{n_splits}Fold_CV.csv", sep=";", index=False)
            performance.to_csv(f"{save_dir}\\results\\performances\\performance_{p}_Group{n_splits}Fold_CV.csv", sep=";",)
            cv_predictions.to_csv(f"{save_dir}\\results\\predictions\\predictions_{p}_core_features_Group{n_splits}Fold_CV.csv", sep=";", index=False)

            if not os.path.isdir(f"{save_dir}\\results\\models\\{p}"):
                os.mkdir(f"{save_dir}\\results\\models\\{p}")

            for k, mod in enumerate(trained_models):
                save_name = str(k) + "_" + mod.split("_")[0] + "_" + mod.split("_")[1] + "_" + mod.split("_")[-1]
                with open(f"{save_dir}\\results\\models\\{p}\\{mod}.pickle", "wb") as save_file:
                    pickle.dump(trained_models[mod], save_file)

            # print("Saved performances, predictions, log files and models !")

        # Feature importance retried by random forest classifier
        # print(f"Bacterial features : Clermont_Phylo, ST_Warwick, LPS_type, O-type, H-type.")
        # print(f"Phage features : Morphotype, Genus, Phage_host.")

        # get best model on test set
        perf_by_model = performance.loc[performance["dataset"] == "test"].groupby("model")["avg_precision"].mean()
        model_name = perf_by_model.sort_values(ascending=False).index[0]

        print(f"Best model: {model_name}")

        clfs = []
        for mod in os.listdir(save_dir + f"\\results\\models\\{p}"):
            if mod.startswith(p + "_" + model_name) and mod.endswith("pickle"):
                clfs.append(pickle.load(open(save_dir +  f"\\results\\models\\{p}\\" + mod, "rb")))

        # save feature importance
        if model_name.startswith("RF"):
            feature_importances = pd.DataFrame([clf.feature_importances_ for clf in clfs], columns=X_oh.columns).melt()
        elif model_name.startswith("LogReg"):
            feature_importances = pd.DataFrame([clf.coef_[0] for clf in clfs], columns=X_oh.columns).melt()
        else:
            continue

        sorted_by_average_importance = feature_importances.groupby("variable").mean().sort_values("value", ascending=False).reset_index().rename({"value": "average_importance"}, axis=1)
        feature_importances = pd.merge(feature_importances, sorted_by_average_importance, on="variable")
        feature_importances["phage"] = p
        feature_importances["model"] = model_name
        feature_importances.to_csv(f"{save_dir}/results/feature_importances/{p}_feature_importance.csv", sep=";", index=False)       
        