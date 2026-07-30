"""
Shared machinery for runners/run_downstream_final.py and
runners/run_downstream_agp_final.py.

Differences from the older mgm/src/downstream_utils.py:

  * Classification only -- no regression, so `age` and `bmi` are not run.
  * Only `linear` and `xgboost` -- random forest and tabpfn are gone.
  * Nested cross-validation: an outer StratifiedKFold estimates
    generalization and a full grid search reruns inside every outer fold on
    that fold's training rows only. The old AGP runner searched once on the
    whole task and then refit the winner per fold, which let every held-out
    fold influence the model that scored it.
  * Features are config-driven ("feature sets"): a source (.X or an .obsm
    key) plus a per-sample normalization.

The parameter grids below are fixed by the project spec -- results are only
comparable across runs if they stay put.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from scipy.stats import t as student_t
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize
from sklearn.utils.class_weight import compute_class_weight

from utils.downstream_eval_utils import evaluate_binary

logger = logging.getLogger("downstream_final")

TARGET_COLNAME = "downstream_task"
CATEGORICAL_LABEL_COL = "categorical_label"
CONTINUOUS_LABEL_COL = "continuous_label"
STUDY_COLNAME = "study_id"


def setup_logging(verbose=False):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")


# ── compute budget ────────────────────────────────────────────────────────────

def allocated_cpus():
    """
    Cores we may actually use. os.cpu_count() reports the whole machine (64
    here) but Slurm usually gives us far fewer (8), and sizing joblib off the
    machine count spawns many more workers than cores.
    """
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    return len(os.sched_getaffinity(0))


def cuda_available():
    """Whether this XGBoost build can really train on a GPU -- probe, don't guess."""
    try:
        from xgboost import XGBClassifier
        XGBClassifier(device="cuda", tree_method="hist", n_estimators=2).fit(
            np.random.rand(32, 4).astype(np.float32), np.random.randint(0, 2, 32))
        return True
    except Exception as exc:
        logger.info("No usable GPU for XGBoost (%s); using CPU.", type(exc).__name__)
        return False


@dataclass
class ExecConfig:
    """
    Where fits run and how wide. Separate from the grids on purpose: these
    change wall-clock, never which model wins.
    """
    n_cpus: int
    device: str          # "cpu" | "cuda"
    search_n_jobs: int   # grid candidates evaluated in parallel
    model_n_jobs: int    # threads inside one XGBoost fit

    @classmethod
    def build(cls, device="auto", n_cpus=None):
        n_cpus = n_cpus or allocated_cpus()
        if device == "auto":
            device = "cuda" if cuda_available() else "cpu"

        if device == "cuda":
            # One GPU, so candidates go through it one at a time; parallelism
            # comes from the GPU itself.
            return cls(n_cpus, device, search_n_jobs=1, model_n_jobs=n_cpus)
        # Small data (<=10k rows x 1514 taxa): many single-threaded fits at
        # once beats one wide fit at a time, and avoids nesting n_jobs=-1
        # inside n_jobs=-1.
        return cls(n_cpus, device, search_n_jobs=n_cpus, model_n_jobs=1)

    def describe(self):
        return (f"cpus={self.n_cpus} device={self.device} "
                f"search_n_jobs={self.search_n_jobs} model_n_jobs={self.model_n_jobs}")


# ── features ──────────────────────────────────────────────────────────────────

# Pseudocount added before any log/closure, matching the previous pipeline.
# ~95% of entries are zero, so this is what keeps clr/rel_ab defined.
PSEUDOCOUNT = 1e-8


def _closure(X):
    """Scale each row to sum to 1 -- skbio.stats.composition.closure."""
    return X / X.sum(axis=1, keepdims=True)


def _clr(X):
    """
    Centred log-ratio, i.e. skbio's clr(closure(X + eps)).

    skbio's clr is log(x) minus the row mean of log(x), and its closure is a
    row-sum rescale, so both are reproduced here in numpy rather than pulling
    in scikit-bio -- it is not installed in mgm_env (only in the base env),
    and this keeps the values identical to the earlier pipeline.
    """
    log_x = np.log(_closure(X + PSEUDOCOUNT))
    return log_x - log_x.mean(axis=1, keepdims=True)


# Names and semantics carried over from the previous normalization script so
# results stay comparable. All are per-sample (each row transformed on its
# own), so they can be applied once up front without leaking between CV
# folds. Anything feature-wise added later (column scaling, quantile
# transforms, batch correction, PCA) must instead go inside the model
# pipeline, next to the StandardScaler, so it is fitted on training rows only.
NORMALIZATIONS = {
    "none": lambda X: X,
    "l2": lambda X: normalize(X, norm="l2", axis=1),
    "clr": _clr,
    "log": lambda X: np.log1p(X),
    "rel_ab": lambda X: _closure(X + PSEUDOCOUNT),
}


@dataclass
class FeatureMatrix:
    """
    Dense sample x feature matrix with a unique sample index.

    Kept as a bare float32 array rather than a DataFrame: half the memory,
    what XGBoost wants anyway, and it sidesteps the taxa-name problem the old
    code worked around with `_to_xgb_safe` (21 taxa contain '[' or ']', which
    XGBoost rejects as feature names).
    """
    values: np.ndarray
    index: pd.Index
    columns: List[str]

    def rows_for(self, sample_ids):
        positions = self.index.get_indexer(sample_ids)
        # get_indexer returns -1 for misses, which would silently fetch the
        # last row instead of failing -- worth the one check.
        if (positions < 0).any():
            raise KeyError(f"{(positions < 0).sum()} sample ID(s) have no feature row, "
                           f"e.g. {list(sample_ids[positions < 0][:5])}")
        return self.values[positions]


def read_obs(h5ad_path):
    """Read .obs without touching X."""
    with h5py.File(h5ad_path, "r") as f:
        return ad.io.read_elem(f["obs"])


def load_features(h5ad_path, spec, obs_names):
    """
    Load one feature set's raw values. Thresholding and normalization happen
    later, in preprocess_features, so they can see every split at once.

    `spec` is {"source": "X"} or {"source": "obsm", "key": ...}.

    The h5ads are "long" -- one row per (sample, task) pair, so a sample ID
    repeats once per task it appears in (22501 rows / 10292 samples in train,
    82786 / 4541 in AGP). Every repeat carries the same profile, so we read
    only the first row per sample; that is also what keeps the 1 GB AGP
    matrix from being fully materialized.
    """
    source = spec.get("source", "X")
    positions = np.flatnonzero(~obs_names.duplicated(keep="first"))

    with h5py.File(h5ad_path, "r") as f:
        if source == "X":
            node, columns = f["X"], list(ad.io.read_elem(f["var"]).index)
        else:
            node = f["obsm"][spec["key"]]
            columns = [f"{spec['key']}_{i}" for i in range(node.shape[1])]
        values = node[positions, :].astype(np.float32)

    logger.info("  %s: %d samples x %d features (raw)",
                os.path.basename(h5ad_path), *values.shape)
    return FeatureMatrix(values, obs_names[positions], columns)


def apply_thresholding(features, prevalence_threshold, abundance_threshold):
    """
    Drop features that are too rare or too low-abundance, on raw counts.

    Prevalence is the fraction of samples where the feature is non-zero and
    abundance is its mean across samples, both computed over every sample in
    the matrix -- i.e. train and test together, as in the previous pipeline.
    That does mean the held-out folds inform which features survive. It is
    unsupervised (labels are never consulted) so the effect is mild and it is
    standard practice for taxa filtering, but it is a real choice: to avoid it
    entirely, the filter would have to be refitted inside each fold.
    """
    X = features.values
    prevalence = (X > 0).mean(axis=0)
    mean_abundance = X.mean(axis=0)
    keep = np.flatnonzero((prevalence >= prevalence_threshold)
                          & (mean_abundance >= abundance_threshold))
    logger.info("  thresholding (prevalence>=%s, abundance>=%s): keeping %d of %d features",
                prevalence_threshold, abundance_threshold, len(keep), X.shape[1])
    if len(keep) == 0:
        raise ValueError("Thresholding removed every feature; loosen the thresholds.")
    return FeatureMatrix(X[:, keep], features.index, [features.columns[i] for i in keep])


def preprocess_features(features, spec):
    """
    Threshold, then normalize -- in that order, so thresholds are judged on
    raw counts rather than on transformed values.

    Thresholds are optional; giving only one leaves the other at 0.0, which
    is a no-op.
    """
    prevalence = spec.get("prevalence_threshold")
    abundance = spec.get("abundance_threshold")
    if prevalence is not None or abundance is not None:
        features = apply_thresholding(features, prevalence or 0.0, abundance or 0.0)

    method = spec.get("normalization", "none")
    values = np.ascontiguousarray(NORMALIZATIONS[method](features.values), dtype=np.float32)
    logger.info("  normalization '%s': %d samples x %d features", method, *values.shape)
    return FeatureMatrix(values, features.index, features.columns)


def concat_features(first, second):
    """
    One lookup table spanning both h5ads, so pooled and presplit tasks alike
    can pull any sample they reference.

    3927 sample IDs appear in both files: the train/test split was drawn
    independently per task, so a sample can be a train row for one task and a
    test row for another. Those duplicates are the same physical sample with
    the same profile, so keep the first and leave the index unique.
    """
    index = first.index.append(second.index)
    values = np.concatenate([first.values, second.values])
    keep = ~index.duplicated(keep="first")
    return FeatureMatrix(values[keep], index[keep], first.columns)


# ── task selection ────────────────────────────────────────────────────────────

@dataclass
class TaskSplit:
    """Sample IDs, labels and studies for one task, per split."""
    name: str
    label_col: str
    sample_ids: Dict[str, pd.Index]
    labels: Dict[str, pd.Series]
    studies: List[str]


def infer_label_col(task_name, obs_frames):
    """
    Which of categorical_label / continuous_label holds this task's labels,
    read off the data rather than declared in the config.
    """
    cols = set()
    for split_name, obs in obs_frames.items():
        rows = obs[obs[TARGET_COLNAME] == task_name]
        has_cat = rows[CATEGORICAL_LABEL_COL].notna().any()
        has_cont = rows[CONTINUOUS_LABEL_COL].notna().any()
        if has_cat and has_cont:
            raise ValueError(f"Task '{task_name}' ({split_name}) has both label types.")
        cols.add(CATEGORICAL_LABEL_COL if has_cat else CONTINUOUS_LABEL_COL)
    if len(cols) != 1:
        raise ValueError(f"Task '{task_name}': label column differs between splits.")
    return cols.pop()


def build_task_split(task_name, obs_frames, ignored_labels):
    label_col = infer_label_col(task_name, obs_frames)
    sample_ids, labels, studies = {}, {}, []

    for split_name, obs in obs_frames.items():
        # obs.index is not unique (a sample repeats once per task), so select
        # positionally with a mask -- .loc[ids] would also pull that sample's
        # rows for other tasks.
        mask = ((obs[TARGET_COLNAME] == task_name)
                & obs[label_col].notna()
                & ~obs[label_col].isin(ignored_labels))
        rows = obs[mask]

        sample_ids[split_name] = rows.index
        # categorical_label is a Categorical carrying every level in the file
        # (28 for HMC, 146 for AGP), so a task's slice would otherwise report
        # dozens of zero-count classes. Dropping to object keeps class
        # counting, weighting and encoding honest.
        labels[split_name] = rows[label_col].astype(object)
        if STUDY_COLNAME in rows:
            studies += rows[STUDY_COLNAME].astype(object).unique().tolist()

    return TaskSplit(task_name, label_col, sample_ids, labels, sorted(set(studies)))


def resolve_split_mode(task, declared="auto"):
    """
    "combined_cv" -- pool the splits and run nested CV (single-study tasks)
    "presplit"    -- train on train, evaluate on test

    Auto-detection counts distinct study_id values. On the current data that
    routes `location` (12 train / 7 test studies) and `sex` (3 / 2) to
    presplit and everything else to combined_cv. Those two have *disjoint*
    train and test studies, so their split is a cross-study holdout and
    reshuffling it into folds would destroy what it measures.
    """
    if declared != "auto":
        return declared
    return "combined_cv" if len(task.studies) <= 1 else "presplit"


def combine_splits(task):
    """
    Pool a single-study task's splits for cross-validation, as one label
    Series indexed by sample ID.

    Within a task the splits are expected to be disjoint; if they are not, the
    same sample would land in both a training fold and its held-out fold.
    (Train and test do share 3927 IDs *overall* -- the split was drawn per
    task, so a sample can be a train row for one task and a test row for
    another -- which is why this is checked per task rather than assumed.)
    """
    labels = pd.concat(list(task.labels.values()))
    if labels.index.duplicated().any():
        raise ValueError(f"Task '{task.name}': sample(s) appear in both splits; "
                         f"pooling would leak them across CV folds.")
    return labels


def drop_rare_classes(labels, min_count, task_name):
    """
    Remove only the classes stratification cannot place in every fold.

    The floor is outer_folds, not outer_folds * inner_folds: a class with
    exactly `outer_folds` members gets one member in each outer test fold and
    keeps outer_folds - 1 in the outer training fold, which the inner
    StratifiedKFold(3) still splits with at least one member per fold. So 5
    members is enough for 5x3 nested CV and nothing above that is filtered.

    Below the floor this is not a quality judgement, it is a hard constraint.
    StratifiedKFold only warns and carries on, leaving some test folds with
    zero members of the class, and then:
      * roc_auc_ovr_weighted raises when a fold's y_true is missing a class
        that y_score has a column for, so inner search scores go NaN;
      * a class with one member is absent from one fold's *training* set, so
        the model returns fewer predict_proba columns than there are classes.
    Keeping such a class produces errors or silently meaningless metrics, not
    a usable-but-noisy estimate.
    """
    counts = labels.value_counts()
    dropped = {str(k): int(v) for k, v in counts[counts < min_count].items()}
    if dropped:
        logger.warning("  '%s': dropping %d class(es) with < %d members, which "
                       "stratification cannot put in every fold: %s",
                       task_name, len(dropped), min_count, dropped)
    return labels[labels.isin(counts[counts >= min_count].index)], dropped


def class_feasibility(labels, min_count):
    """What the class filter will do, for the plan printed before a run."""
    counts = labels.value_counts()
    usable = int((counts >= min_count).sum())
    return usable, len(counts), usable >= 2


# ── models (grids fixed by project spec) ──────────────────────────────────────

def train_xgboost(X_train, y_train, search_type="none", sample_weights=None,
                  inner_cv=3, device="cpu", search_n_jobs=-1, model_n_jobs=-1, seed=42):
    """Returns (best_params, best_model, inner_cv_std)."""
    from xgboost import XGBClassifier

    logger.info("  Training XGBoost on %s: X=%s y=%s", device, X_train.shape, y_train.shape)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    n_classes = len(np.unique(y_train))
    model = XGBClassifier(
        random_state=seed,
        n_jobs=model_n_jobs,
        tree_method="hist",
        device=device,
        eval_metric="auc" if n_classes == 2 else "mlogloss"
    )
    search_scoring = 'roc_auc' if n_classes == 2 else 'roc_auc_ovr_weighted'
    cv = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=seed)

    if search_type == "grid":
        param_grid = {
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 6, 9],
            "n_estimators": [200, 500, 1000],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0]
        }
        search = GridSearchCV(estimator=model, param_grid=param_grid,
                              scoring=search_scoring, cv=cv, n_jobs=search_n_jobs)
    elif search_type == "random":
        param_distributions = {
            "learning_rate": loguniform(1e-3, 3e-1),
            "max_depth": randint(3, 12),
            "n_estimators": randint(100, 1000),
            "subsample": uniform(0.5, 0.5),
            "colsample_bytree": uniform(0.5, 0.5),
            "min_child_weight": loguniform(1e-1, 1e2),
            "gamma": loguniform(1e-8, 1e1),
            "reg_alpha": loguniform(1e-8, 1e1),
            "reg_lambda": loguniform(1e-3, 1e2),
        }
        search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions,
                                    n_iter=200, scoring=search_scoring, cv=cv,
                                    n_jobs=search_n_jobs, random_state=seed)
    elif search_type == "none":
        params = {"learning_rate": 0.05, "max_depth": 7, "n_estimators": 500,
                  "subsample": 1.0, "colsample_bytree": 1.0, "tree_method": "hist"}
        logger.info("    Fixed parameters: %s", params)
        model.set_params(**params)
        model.fit(X_train, y_train_encoded, sample_weight=sample_weights)
        model._label_encoder = label_encoder
        return params, model, None
    else:
        raise ValueError(f"search_type must be 'grid', 'random', or 'none': {search_type}")

    start = time.time()
    search.fit(X_train, y_train_encoded, sample_weight=sample_weights)
    logger.info("    Search: %d candidates x %d folds in %.1fs",
                len(search.cv_results_["params"]), inner_cv, time.time() - start)

    best_model = search.best_estimator_
    best_model._label_encoder = label_encoder
    return (search.best_params_, best_model,
            float(search.cv_results_["std_test_score"][search.best_index_]))


def train_linear(X_train, y_train, search_type="none", sample_weights=None,
                 inner_cv=3, search_n_jobs=-1, seed=42):
    """Returns (best_params, best_model, inner_cv_std)."""
    logger.info("  Training Linear: X=%s y=%s", X_train.shape, y_train.shape)

    n_classes = len(np.unique(y_train))
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(penalty='l1', solver='saga', random_state=seed))
    ])
    param_distributions = {
        'model__max_iter': [500, 1000, 2000, 5000],
        'model__C': 1.0 / np.logspace(-4, 2, 10)
    }
    param_grid = {
        'model__max_iter': [2000],
        'model__C': 1.0 / np.logspace(-3, 1, 5)
    }
    search_scoring = 'roc_auc' if n_classes == 2 else 'f1_weighted'
    cv = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=seed)

    if search_type == "grid":
        search = GridSearchCV(model, param_grid, cv=cv, scoring=search_scoring,
                              n_jobs=search_n_jobs)
    elif search_type == "random":
        search = RandomizedSearchCV(model, param_distributions, cv=cv, scoring=search_scoring,
                                    n_jobs=search_n_jobs, n_iter=5, random_state=seed)
    elif search_type == "none":
        params = {"model__C": 1.0}
        logger.info("    Fixed parameters: %s", params)
        model.set_params(**params)
        model.fit(X_train, y_train, model__sample_weight=sample_weights)
        return params, model, None
    else:
        raise ValueError(f"search_type must be 'grid', 'random', or 'none': {search_type}")

    start = time.time()
    search.fit(X_train, y_train, model__sample_weight=sample_weights)
    logger.info("    Search: %d candidates x %d folds in %.1fs",
                len(search.cv_results_["params"]), inner_cv, time.time() - start)
    return (search.best_params_, search.best_estimator_,
            float(search.cv_results_["std_test_score"][search.best_index_]))


def fit_method(method, X_train, y_train, search_type, weights, inner_cv, exec_conf, seed):
    if method == "xgboost":
        return train_xgboost(X_train, y_train, search_type, weights, inner_cv,
                             exec_conf.device, exec_conf.search_n_jobs,
                             exec_conf.model_n_jobs, seed)
    return train_linear(X_train, y_train, search_type, weights, inner_cv,
                        exec_conf.n_cpus, seed)


def balanced_weights(y):
    classes = np.unique(y)
    weights = dict(zip(classes, compute_class_weight("balanced", classes=classes, y=y)))
    return np.array([weights[v] for v in y], dtype=float)


def predict_proba(model, X, classes):
    """
    predict_proba with columns forced into `classes` order.

    The two methods label their columns differently -- the linear pipeline's
    follow the raw label strings, XGBoost's follow the LabelEncoder it fitted
    internally -- so reindex to keep per-class metrics attached to the right
    class in every fold.
    """
    probs = model.predict_proba(X)
    encoder = getattr(model, "_label_encoder", None)
    model_classes = list(encoder.classes_) if encoder is not None else list(model.classes_)
    if model_classes == list(classes):
        return probs
    position = {c: i for i, c in enumerate(model_classes)}
    return probs[:, [position[c] for c in classes]]


# ── evaluation ────────────────────────────────────────────────────────────────

def score(probs, y_true, classes, out_dir):
    """
    Per-class one-vs-rest metrics via this repo's evaluate_binary, support-
    weighted into the 4-key summary ResultsTable/ResultsWriter expect.
    `probs` are already probabilities, `y_true` are integer class indices
    into `classes`.
    """
    y_pred = np.argmax(probs, axis=1)
    rows = {}
    for i, c in enumerate(classes):
        y_true_bin = (y_true == i).astype(int)
        if y_true_bin.sum() == 0:
            continue
        y_pred_bin = (y_pred == i).astype(int)
        row = evaluate_binary(str(c), probs[:, i], y_pred_bin, y_true_bin)
        row["F1"] = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        rows[str(c)] = row

    table = pd.DataFrame.from_dict(rows, orient="index")
    weights = table["n_samples"]
    weighted = lambda col: np.average(table[col], weights=weights)

    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "metrics.csv"))

    return {
        "AUROC": weighted("AUC (ROC)"),
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": weighted("F1"),
        "PR-AUC": weighted("Average Precision"),
    }


def aggregate_folds(fold_metrics, confidence=0.95):
    """
    Mean and error margin per metric across outer folds.

    std uses ddof=1 (the folds are a sample of possible splits; the old AGP
    runner used ddof=0, so its numbers ran slightly smaller) and the margin is
    a t-based CI on the mean, which with 5 folds is meaningfully wider than
    1.96 * sem.
    """
    out = {}
    n = len(fold_metrics)
    for key in fold_metrics[0]:
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        mean, std = float(values.mean()), float(values.std(ddof=1))
        margin = float(student_t.ppf(0.5 + confidence / 2, n - 1) * std / np.sqrt(n))
        out[key] = {"mean": mean, "std": std, "ci_margin": margin,
                    "n_folds": n, "values": values.tolist()}
    return out


def run_nested_cv(method, method_conf, features, labels,
                  cv_conf, exec_conf, seed, out_dir, task_name):
    """
    Outer StratifiedKFold for the estimate, a fresh full grid search inside
    each outer fold on that fold's training rows only. Nothing about an outer
    fold's test rows reaches the search that produces the model scoring them.

    Rare classes are dropped up front and the split is stratified, so every
    class appears in every fold and one shared encoding keeps fold metrics
    comparable.
    """
    labels, dropped = drop_rare_classes(labels, cv_conf["min_samples_per_class"], task_name)
    if labels.nunique() < 2:
        logger.warning("  '%s': fewer than 2 usable classes, skipping.", task_name)
        return None

    X = features.rows_for(labels.index)
    y = labels.to_numpy(dtype=object)
    classes = sorted(set(y), key=str)
    class_index = {c: i for i, c in enumerate(classes)}

    outer_folds, inner_folds = cv_conf["outer_folds"], cv_conf["inner_folds"]
    logger.info("  %s: nested CV %dx%d, n=%d, classes=%s",
                method, outer_folds, inner_folds, len(y), classes)

    outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    fold_metrics, fold_params = [], []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y)):
        y_train, y_test = y[train_idx], y[test_idx]
        # Weights are balanced against the training fold only; the search
        # slices this array down to each inner training split.
        weights = balanced_weights(y_train)

        params, model, _ = fit_method(method, X[train_idx], y_train,
                                      method_conf.get("search_type", "grid"),
                                      weights, inner_folds, exec_conf, seed)

        probs = predict_proba(model, X[test_idx], classes)
        y_true = np.array([class_index[v] for v in y_test])
        metrics = score(probs, y_true, classes, os.path.join(out_dir, f"fold_{fold}"))

        fold_metrics.append(metrics)
        fold_params.append(params)
        logger.info("    fold %d/%d (n_test=%d): %s", fold + 1, outer_folds, len(test_idx),
                    {k: round(v, 4) for k, v in metrics.items()})

    aggregates = aggregate_folds(fold_metrics)
    logger.info("  %s: %s", method,
                {k: f"{v['mean']:.4f} +/- {v['ci_margin']:.4f}" for k, v in aggregates.items()})

    return {"eval_mode": "nested_cv", "aggregates": aggregates, "fold_params": fold_params,
            "n_samples": len(y), "classes": [str(c) for c in classes],
            "dropped_classes": dropped}


def run_presplit(method, method_conf, features, train_labels, test_labels,
                 cv_conf, exec_conf, seed, out_dir, task_name):
    """
    Multi-study evaluation: full grid search over `inner_folds` on the train
    split, then one evaluation on the held-out test split.

    No outer CV here, deliberately -- `location` and `sex` are split so their
    train and test studies are disjoint, and reshuffling those rows into
    folds would destroy the cross-study generalization the number measures.
    The cost is that there is no across-fold spread to quote, so the margin
    recorded is the inner search's CV std at the chosen hyperparameters, a
    different quantity flagged by eval_mode="presplit".
    """
    inner_folds = cv_conf["inner_folds"]
    train_labels, dropped = drop_rare_classes(train_labels,
                                              cv_conf["min_samples_per_class"], task_name)
    if train_labels.nunique() < 2:
        logger.warning("  '%s': fewer than 2 usable train classes, skipping.", task_name)
        return None

    # A class never seen in training cannot be predicted.
    unseen = set(test_labels.unique()) - set(train_labels.unique())
    if unseen:
        logger.warning("  Dropping test samples with unseen labels: %s", unseen)
        test_labels = test_labels[~test_labels.isin(unseen)]

    y_train = train_labels.to_numpy(dtype=object)
    y_test = test_labels.to_numpy(dtype=object)
    classes = sorted(set(y_train), key=str)
    class_index = {c: i for i, c in enumerate(classes)}

    logger.info("  %s: presplit, %d-fold search on train, n_train=%d n_test=%d, classes=%s",
                method, inner_folds, len(y_train), len(y_test), classes)

    params, model, inner_std = fit_method(
        method, features.rows_for(train_labels.index), y_train,
        method_conf.get("search_type", "grid"), balanced_weights(y_train),
        inner_folds, exec_conf, seed)

    probs = predict_proba(model, features.rows_for(test_labels.index), classes)
    y_true = np.array([class_index[v] for v in y_test])
    metrics = score(probs, y_true, classes, out_dir)
    logger.info("  %s: %s", method, {k: round(v, 4) for k, v in metrics.items()})

    # The search optimized one scoring function, so its std belongs to that
    # metric alone rather than smeared across all four.
    searched = "AUROC" if len(classes) == 2 else "F1"
    aggregates = {name: {"mean": value, "std": 0.0, "ci_margin": 0.0,
                         "n_folds": 1, "values": [value]}
                  for name, value in metrics.items()}
    if inner_std is not None:
        aggregates[searched].update(std=inner_std, ci_margin=inner_std)

    return {"eval_mode": "presplit", "aggregates": aggregates, "fold_params": [params],
            "n_samples": len(y_train) + len(y_test), "classes": [str(c) for c in classes],
            "dropped_classes": dropped}


# ── results ───────────────────────────────────────────────────────────────────

# Column order for the readable tables; anything else lands after these.
METRIC_ORDER = ["AUROC", "Accuracy", "F1", "PR-AUC"]


def summary_rows(feature_set, method, task, payload):
    """One tidy row per metric. The single definition of a summary row."""
    return [{
        "task": task, "feature_set": feature_set, "method": method,
        "eval_mode": payload["eval_mode"], "metric": metric,
        "mean": agg["mean"], "std": agg["std"], "ci_margin": agg["ci_margin"],
        "ci_low": agg["mean"] - agg["ci_margin"],
        "ci_high": agg["mean"] + agg["ci_margin"],
        "estimate": f"{agg['mean']:.4f} +/- {agg['ci_margin']:.4f}",
        "n_folds": agg["n_folds"], "n_samples": payload["n_samples"],
        "n_classes": len(payload["classes"]),
        "classes": "|".join(payload["classes"]),
        "dropped_classes": json.dumps(payload["dropped_classes"]),
    } for metric, agg in payload["aggregates"].items()]


def fold_rows(feature_set, method, task, payload):
    """One row per outer fold per metric -- where the per-fold spread survives."""
    return [{
        "task": task, "feature_set": feature_set, "method": method,
        "metric": metric, "fold": fold, "value": value,
        "best_params": json.dumps(payload["fold_params"][fold], default=str),
    } for metric, agg in payload["aggregates"].items()
      for fold, value in enumerate(agg["values"])]


def _ordered_metrics(metrics):
    known = [m for m in METRIC_ORDER if m in metrics]
    return known + sorted(m for m in metrics if m not in METRIC_ORDER)


def read_task_results(task_dir):
    """
    Every finished (feature_set, method, payload) under one task directory.

    Read off disk rather than from whatever this process happens to hold, so
    a job that ran a subset of feature sets still writes a summary covering
    everything finished so far instead of overwriting it with its own slice.
    """
    records = []
    if not os.path.isdir(task_dir):
        return records
    for feature_set in sorted(os.listdir(task_dir)):
        fs_dir = os.path.join(task_dir, feature_set)
        if not os.path.isdir(fs_dir):
            continue
        for method in sorted(os.listdir(fs_dir)):
            path = os.path.join(fs_dir, method, "result.json")
            if os.path.exists(path):
                with open(path) as f:
                    records.append((feature_set, method, json.load(f)))
    return records


def write_task_summary(task_dir, task_name):
    """
    Write one task's results where the task lives, so a job that ran only
    this task leaves a complete, readable answer behind.

    Produces summary.csv (tidy, one row per metric) and summary.txt (a table
    of "mean +/- margin" with feature sets and methods down the side).
    """
    rows = [r for fs, method, payload in read_task_results(task_dir)
            for r in summary_rows(fs, method, task_name, payload)]
    if not rows:
        return

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(task_dir, "summary.csv"), index=False)

    # pivot, not pivot_table: (feature_set, method, metric) is unique here, so
    # there is nothing to aggregate, and pivot_table can drop non-numeric
    # values -- these cells are formatted strings.
    table = df.pivot(index=["feature_set", "method"], columns="metric",
                     values="estimate")
    table = table[_ordered_metrics(list(table.columns))]

    head = rows[0]
    dropped = json.loads(head["dropped_classes"])
    lines = [
        f"task:       {task_name}",
        f"eval mode:  {head['eval_mode']}  ({head['n_folds']} fold(s))",
        f"samples:    {head['n_samples']}",
        f"classes:    {head['n_classes']}  ({head['classes'].replace('|', ', ')})",
    ]
    if dropped:
        lines.append(f"dropped:    {dropped}  (too few members to stratify)")
    lines += ["", "values are mean +/- 95% CI margin across folds", "",
              table.to_string(), ""]

    with open(os.path.join(task_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines))
    logger.info("  wrote %s", os.path.join(task_dir, "summary.txt"))


class ResultsTable:
    """
    Accumulates per-task metrics for one experiment into a single table: rows
    are (row_key..., metric) tuples -- e.g. (method, "AUROC") or
    (embed_name, method, "AUROC") -- and columns are task names. Metrics that
    don't apply to a given task (e.g. regression metrics on a classification
    task) are simply never added for that column, so they show up blank.

    When a metric's standard deviation is available (cross-validation, repeated
    runs, ...) it's stored as a sibling row named "<metric>_std".

    row_key must be a tuple of the same length across every `add()` call for
    a given table (e.g. always `(method,)`, or always `(embed_name, method)`);
    pass `()` if there's no extra row level (one model per task, as in the
    finetuning runners).
    """

    def __init__(self, experiment_name, row_key_names=(), results_root="results"):
        self.experiment_name = experiment_name
        self.row_key_names = tuple(row_key_names)
        self.results_root = results_root
        self._rows = {}

    def add(self, row_key, task_name, metrics, std=None):
        row_key = tuple(row_key)
        if len(row_key) != len(self.row_key_names):
            raise ValueError(f"row_key {row_key} doesn't match row_key_names {self.row_key_names}")
        for metric_name, value in metrics.items():
            self._rows.setdefault(row_key + (metric_name,), {})[task_name] = value
            std_value = std.get(metric_name) if std else None
            if std_value is not None:
                self._rows.setdefault(row_key + (f"{metric_name}_std",), {})[task_name] = std_value

    def save(self):
        os.makedirs(self.results_root, exist_ok=True)
        path = os.path.join(self.results_root, f"{self.experiment_name}.csv")

        if not self._rows:
            print(f"\t No results recorded for experiment '{self.experiment_name}', skipping save.")
            return None

        df = pd.DataFrame.from_dict(self._rows, orient="index")
        index_names = list(self.row_key_names) + ["metric"]
        if len(index_names) == 1:
            df.index = pd.Index([k[0] for k in df.index], name=index_names[0])
        else:
            df.index = pd.MultiIndex.from_tuples(df.index, names=index_names)
        df = df.sort_index(axis=0).sort_index(axis=1)

        df.to_csv(path)
        print(f"\nResults table for experiment '{self.experiment_name}' -> {path}")
        print(df.to_string())
        return df


class ResultsWriter:
    """Accumulates tidy summary and per-fold rows across a whole run."""

    def __init__(self, experiment_name, results_root="results"):
        self.experiment_name = experiment_name
        self.results_root = results_root
        self.folds, self.summary = [], []

    def add(self, feature_set, method, task, payload):
        self.summary += summary_rows(feature_set, method, task, payload)
        self.folds += fold_rows(feature_set, method, task, payload)

    def save(self):
        os.makedirs(self.results_root, exist_ok=True)
        for rows, suffix in [(self.folds, "folds"), (self.summary, "summary")]:
            if rows:
                path = os.path.join(self.results_root, f"{self.experiment_name}_{suffix}.csv")
                pd.DataFrame(rows).to_csv(path, index=False)
                logger.info("Wrote %s", path)


def record_result(results, writer, feature_set, method, task, payload):
    """Add one finished unit of work to both the wide and tidy tables."""
    aggregates = payload["aggregates"]
    results.add((feature_set, method), task,
                {k: v["mean"] for k, v in aggregates.items()},
                {k: v["std"] for k, v in aggregates.items() if v["std"]} or None)
    writer.add(feature_set, method, task, payload)


def collect_results(output_root, experiment_name, results_root):
    """
    Rebuild the run-wide tables by walking every result.json under
    output_root, without refitting anything.

    This is the epilogue for a Slurm job array: each array job runs one task
    and writes its own directory, so the run-wide tables cannot be written
    safely by any single job (they would all race for the same path). Run
    this once after the array finishes.
    """
    results = ResultsTable(experiment_name, row_key_names=("feature_set", "method"),
                           results_root=results_root)
    writer = ResultsWriter(experiment_name, results_root=results_root)

    if not os.path.isdir(output_root):
        logger.warning("Nothing to collect: %s does not exist.", output_root)
        return

    found = 0
    for task in sorted(os.listdir(output_root)):
        task_dir = os.path.join(output_root, task)
        for feature_set, method, payload in read_task_results(task_dir):
            record_result(results, writer, feature_set, method, task, payload)
            found += 1

    logger.info("Collected %d result(s) from %s", found, output_root)
    results.save()
    writer.save()


def load_cached(path, fingerprint):
    """Reuse a finished unit of work only if the settings behind it still match."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        payload = json.load(f)
    if payload.get("fingerprint") != fingerprint:
        logger.info("  Cached result is stale, recomputing.")
        return None
    return payload


def save_cached(path, fingerprint, payload):
    with open(path, "w") as f:
        json.dump({"fingerprint": fingerprint, **payload}, f, indent=2, default=str)


# ── config ────────────────────────────────────────────────────────────────────

def load_config(path):
    """
    Read a run config and fill in defaults. See configs/downstream_final.json
    for the full shape and per-key notes.
    """
    with open(path) as f:
        config = json.load(f)

    config.setdefault("seed", 42)
    config.setdefault("feature_sets", {"raw": {"source": "X", "normalization": "none"}})
    config["cv"] = {"outer_folds": 5, "inner_folds": 3, **config.get("cv", {})}
    config["execution"] = {"device": "auto", "n_cpus": None, **config.get("execution", {})}

    # Default to the stratification floor: a class needs one member per outer
    # test fold. See drop_rare_classes for why anything below that breaks
    # rather than merely degrades.
    config["cv"].setdefault("min_samples_per_class", config["cv"]["outer_folds"])
    if config["cv"]["min_samples_per_class"] < config["cv"]["outer_folds"]:
        raise ValueError(
            f"cv.min_samples_per_class={config['cv']['min_samples_per_class']} is below "
            f"outer_folds={config['cv']['outer_folds']}; stratification cannot place such "
            f"a class in every fold.")

    defaults = config.get("defaults", {}).get("methods", {})
    for name, task in config["tasks"].items():
        task.setdefault("ignored_labels", [])
        task.setdefault("split_mode", "auto")
        # Per-task escape hatch: a small task can be worth running at a lower
        # floor (with a wider margin) rather than dropped entirely.
        task.setdefault("min_samples_per_class", config["cv"]["min_samples_per_class"])
        if not task.get("methods"):
            task["methods"] = json.loads(json.dumps(defaults))
        for method in task["methods"]:
            if method not in ("linear", "xgboost"):
                raise ValueError(f"Task '{name}': unsupported method '{method}'.")
    return config


def cv_for(config, task_conf):
    """cv settings in force for one task, after its per-task override."""
    return {**config["cv"], "min_samples_per_class": task_conf["min_samples_per_class"]}


def grid_size(method, search_type):
    """Candidate count, for the cost estimate printed before a run."""
    if search_type == "none":
        return 1
    if search_type == "random":
        return 200 if method == "xgboost" else 5
    return 2 * 3 * 3 * 2 * 2 if method == "xgboost" else 5   # 72 / 5


def fit_count(cv_conf, methods, mode):
    """Model fits one task costs, per feature set."""
    total = 0
    for method, conf in methods.items():
        candidates = grid_size(method, conf.get("search_type", "grid")) * cv_conf["inner_folds"]
        total += (cv_conf["outer_folds"] * (candidates + 1) if mode == "combined_cv"
                  else candidates + 1)
    return total
