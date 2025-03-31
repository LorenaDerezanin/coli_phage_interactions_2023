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

# Get the repository root directory
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Set up save directory
save_dir = os.path.join(repo_root, "dev", "predictions", "results")
os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "feature_importances"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "performances"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "predictions"), exist_ok=True)

# Load data
interaction_matrix = pd.read_csv(os.path.join(repo_root, "data", "interactions", "interaction_matrix.csv"), sep=";").set_index("bacteria")
print("Unique values in raw interaction matrix:", interaction_matrix.values.ravel())

# Convert numeric values to binary classification
# Values 0.0 and 1.0 are considered negative interactions (0)
# Values 2.0, 3.0, and 4.0 are considered positive interactions (1)
interaction_matrix = interaction_matrix.replace({0.0: 0, 1.0: 0, 2.0: 1, 3.0: 1, 4.0: 1})
print("Unique values after binary conversion:", pd.unique(interaction_matrix.values.ravel()))

# interaction_matrix = interaction_matrix.loc[interaction_matrix.count(axis=1)[interaction_matrix.count(axis=1) > 70].index].fillna(0)

phage_feat_names = ["Morphotype", "Genus", "Phage_host"]
print(f"Phage features : {phage_feat_names}")

# Load features using the correct file paths
phage_features = pd.read_csv(os.path.join(repo_root, "data", "genomics", "phages", "guelin_collection.csv"), sep=";").set_index("phage").loc[interaction_matrix.columns, phage_feat_names]
bact_features = pd.read_csv(os.path.join(repo_root, "data", "genomics", "bacteria", "picard_collection.csv"), sep=";").set_index("bacteria")

# Create cross-validation groups based on bacteria phylogroups
cv_clusters = bact_features[["Clermont_Phylo"]].copy()
cv_clusters.columns = ["cluster"]

bact_embeddings = pd.read_csv(os.path.join(repo_root, "dev", "predictions", "core_genome", "UMAP_dim_reduction_from_phylogeny", "data", "coli_umap_8_dims.tsv"), sep="\t").set_index("bacteria")

bact_features = pd.merge(bact_features, bact_embeddings, left_index=True, right_index=True)

bact_feat_names = "(UMAP|O-type|LPS|ST_Warwick|Klebs|ABC_serotype)"
bact_features = bact_features.filter(regex=bact_feat_names, axis=1)

def preprocess_interaction_matrix(interaction_matrix):
    """Filter out phages with single-class interactions and generate a detailed report."""
    original_count = len(interaction_matrix.columns)
    valid_phages = []
    excluded_phages = []
    
    # Create a detailed report of excluded phages
    excluded_report = {
        'total_phages': original_count,
        'excluded_phages': [],
        'summary': {
            'all_negative': 0,
            'all_positive': 0,
            'interaction_counts': {}
        }
    }
    
    for phage in interaction_matrix.columns:
        unique_values = np.unique(interaction_matrix[phage])
        interaction_count = sum(interaction_matrix[phage] == unique_values[0])
        
        if len(unique_values) > 1:  # Only keep phages with both positive and negative interactions
            valid_phages.append(phage)
        else:
            excluded_phages.append({
                'phage': phage,
                'class': unique_values[0],
                'count': interaction_count,
                'percentage': (interaction_count / len(interaction_matrix)) * 100
            })
            
            # Update summary statistics
            if unique_values[0] == 0:
                excluded_report['summary']['all_negative'] += 1
            else:
                excluded_report['summary']['all_positive'] += 1
            
            # Track interaction counts
            excluded_report['summary']['interaction_counts'][phage] = interaction_count
    
    # Create a report of excluded phages
    excluded_df = pd.DataFrame(excluded_phages)
    if not excluded_df.empty:
        # Save detailed CSV report
        excluded_df.to_csv(os.path.join(save_dir, "excluded_phages.csv"), index=False)
        
        # Save summary report as JSON
        with open(os.path.join(save_dir, "excluded_phages_summary.json"), 'w') as f:
            json.dump(excluded_report, f, indent=4)
        
        # Print summary
        print(f"\nPhage Interaction Analysis:")
        print(f"Total phages: {original_count}")
        print(f"Excluded phages: {len(excluded_phages)}")
        print(f"Valid phages: {len(valid_phages)}")
        print(f"\nExcluded Phages Breakdown:")
        print(f"- Phages with only negative interactions: {excluded_report['summary']['all_negative']}")
        print(f"- Phages with only positive interactions: {excluded_report['summary']['all_positive']}")
        
        # Print top 5 phages with most interactions
        print("\nTop 5 excluded phages by number of interactions:")
        top_5 = sorted(excluded_report['summary']['interaction_counts'].items(), 
                      key=lambda x: x[1], reverse=True)[:5]
        for phage, count in top_5:
            print(f"- {phage}: {count} interactions")
    
    return interaction_matrix[valid_phages]

# In the main code:
print("\nPreprocessing interaction matrix...")
interaction_matrix = preprocess_interaction_matrix(interaction_matrix)
print(f"Processing {len(interaction_matrix.columns)} phages with both positive and negative interactions")

for p in interaction_matrix.columns:
    print(f"\nProcessing phage {p}...")

    # Filter phages according to phylogeny
    phage_feat = phage_features.loc[[p]]
    interaction_mat = interaction_matrix[[p]]

    phage_feat = phage_feat.drop(["Morphotype", "Genus"], axis=1)

    # wide to long
    interaction_matrix_long = interaction_mat.unstack().reset_index().rename({"level_0": "phage", 0: "y"}, axis=1).sort_values(["bacteria", "phage"])  # force row order

    # Add the cross-validation index of each observation for Leave-one-strain-out CV
    interaction_matrix_long = pd.merge(interaction_matrix_long, cv_clusters, left_on=["bacteria"], right_index=True).set_index("cluster")
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
        aliases = {
            LogisticRegression: "LogReg", 
            RandomForestClassifier: "RF", 
            DummyClassifier: "Dummy", 
            MLPClassifier: "MLP",
            BernoulliNB: "NaiveBayes", 
            DecisionTreeClassifier: "DecTree",
            GradientBoostingClassifier: "GBoost"
        }
        name = aliases[type(model)]
        if type(model) == LogisticRegression:
            name += "_" + model.penalty
        elif type(model) == RandomForestClassifier:
            name += "_" + str(model.n_estimators) + "_" + str(model.max_depth)
        elif type(model) == GradientBoostingClassifier:
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

    def perform_group_cross_validation(X, y, n_splits=5, groups=None, model_list=None, index_names=None, do_scale=False):
        if model_list is None:
            model_list = [
                ("LogReg_l2_weight=3", LogisticRegression(C=1/3, max_iter=1000)),
                ("LogReg_l2_weight=1", LogisticRegression(C=1, max_iter=1000)),
                ("LogReg_l2_weight=0.3", LogisticRegression(C=1/0.3, max_iter=1000)),
                ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=5)),
                ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, max_depth=3)),
                ("MLP", MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)),
                ("DecisionTree", DecisionTreeClassifier(max_depth=5)),
                ("Dummy", DummyClassifier(strategy="most_frequent")),
                ("BernoulliNB", BernoulliNB())
            ]
        
        predictions = []
        logs = []
        performance = []
        trained_models = []
        
        # Check if we have both classes in the data
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print(f"Warning: Only one class ({unique_classes[0]}) found in the data. Skipping cross-validation.")
            return [], [], pd.DataFrame(), []
        
        group_kfold = GroupKFold(n_splits=n_splits)
        umap_dim = X.shape[1] // 2
        std_scaler = StandardScaler()
        if do_scale:
            std_scaler.fit(X)

        for i, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups)):
            X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
            
            # Check for single class in training data
            unique_train_classes = np.unique(y_train)
            if len(unique_train_classes) < 2:
                print(f"Warning: Only one class ({unique_train_classes[0]}) found in training data for fold {i}. Skipping this fold.")
                continue
            
            if do_scale:
                X_train = std_scaler.transform(X_train)
                X_test = std_scaler.transform(X_test)
            
            chosen_groups = groups[train_idx]
            assert(set(X_train.index).intersection(set(X_test.index)) == set())
            
            for model_name, model in model_list:
                alias = get_alias(model)
                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Metrics
                    if np.unique(y_test).shape[0] > 1:  # Cannot compute metrics if only one class is predicted
                        cm = confusion_matrix(y_test, y_pred)
                        if cm.shape == (2, 2):
                            tn, fp, fn, tp = cm.ravel()
                        else:
                            # Handle single-class case
                            if len(np.unique(y_test)) == 1:
                                if np.unique(y_test)[0] == 0:
                                    tn, fp, fn, tp = len(y_test), 0, 0, 0
                                else:
                                    tn, fp, fn, tp = 0, 0, 0, len(y_test)
                            else:
                                tn, fp, fn, tp = 0, 0, 0, 0
                        precision, recall, f1 = precision_score(y_test, y_pred, pos_label=1), recall_score(y_test, y_pred, pos_label=1), f1_score(y_test, y_pred, pos_label=1)
                        try:
                            average_prec = average_precision_score(y_test, y_pred_proba)
                            roc_auc = roc_auc_score(y_test, y_pred_proba)
                        except (ValueError, IndexError):
                            # Handle cases where we can't compute these metrics
                            average_prec = np.nan
                            roc_auc = np.nan

                        performance.append({"model": alias, "fold": i, "dataset": "test", "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc, "avg_precision": average_prec,
                                            "tp": tp, "fp": fp, "tn": tn, "fn": fn,})
                    else:
                        performance.append({"model": np.nan, "fold": np.nan, "dataset": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "roc_auc": np.nan, "avg_precision": np.nan,
                                            "tp": np.nan, "fp": np.nan, "tn": np.nan, "fn": np.nan,})

                    # Collect predictions (test set only)
                    if alias != "Dummy":
                        preds = index_names.iloc[test_idx].copy()
                    else:
                        preds = index_names.iloc[train_idx].copy()
                    preds["y_pred"] = y_pred
                    preds["y_pred_proba"] = y_pred_proba
                    preds["fold"] = i
                    preds["model"] = alias
                    preds["dataset"] = "test"
                    predictions.append(preds)  # add bacteria-phage name as index instead of integer (avoid ambiguity)

                except (NotFittedError, ValueError) as e:
                    print(f"Error fitting model {alias}: {e}")
                    performance.append({"model": alias, "fold": i, "dataset": "test", "precision": np.nan, "recall": np.nan, "f1": np.nan, "roc_auc": np.nan, "avg_precision": np.nan,
                                        "tp": np.nan, "fp": np.nan, "tn": np.nan, "fn": np.nan,})

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
        logs, performance, cv_predictions, trained_models = perform_group_cross_validation(
            X_oh, 
            y, 
            n_splits=n_splits, 
            groups=interaction_with_features.index, 
            index_names=bact_phage_names,
            do_scale=False
        )
        
        performance["phage"] = p
        cv_predictions["phage"] = p

        performance = performance.set_index("phage")
        cv_predictions = cv_predictions.set_index("phage")
        
        cv_predictions = pd.merge(cv_predictions, interaction_with_features[["bacteria", "phage", "y"]], on=["bacteria", "phage"])  # add real interaction values

        overwrite_files = True  # overwrite log and performance files
        if overwrite_files:
            logs.to_csv(os.path.join(save_dir, "logs", f"logs_{p}_Group{n_splits}Fold_CV.csv"), sep=";", index=False)
            performance.to_csv(os.path.join(save_dir, "performances", f"performance_{p}_Group{n_splits}Fold_CV.csv"), sep=";",)
            cv_predictions.to_csv(os.path.join(save_dir, "predictions", f"predictions_{p}_core_features_Group{n_splits}Fold_CV.csv"), sep=";", index=False)

            if not os.path.isdir(os.path.join(save_dir, "models", p)):
                os.mkdir(os.path.join(save_dir, "models", p))

            for k, (model_name, model) in enumerate(trained_models):
                save_name = f"{k}_{model_name}"
                with open(os.path.join(save_dir, "models", p, f"{save_name}.pickle"), "wb") as save_file:
                    pickle.dump(model, save_file)

            # print("Saved performances, predictions, log files and models !")

        # Feature importance retried by random forest classifier
        # print(f"Bacterial features : Clermont_Phylo, ST_Warwick, LPS_type, O-type, H-type.")
        # print(f"Phage features : Morphotype, Genus, Phage_host.")

        # get best model on test set
        perf_by_model = performance.loc[performance["dataset"] == "test"].groupby("model")["avg_precision"].mean()
        model_name = perf_by_model.sort_values(ascending=False).index[0]

        print(f"Best model: {model_name}")

        clfs = []
        for mod in os.listdir(os.path.join(save_dir, "models", p)):
            if mod.startswith(p + "_" + model_name) and mod.endswith("pickle"):
                clfs.append(pickle.load(open(os.path.join(save_dir, "models", p, mod), "rb")))

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
        feature_importances.to_csv(os.path.join(save_dir, "feature_importances", f"{p}_feature_importance.csv"), sep=";", index=False)       
        