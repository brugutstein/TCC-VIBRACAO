from __future__ import annotations

import os
import time
import pickle
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score,
)

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import Adam

from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report,confusion_matrix

import time
from rich.console import Console
from rich.table import Table

console = Console()

# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = Path(r"F:\TCC\Dataset\NLM-TCC\pump_fault_tcc")

CACHE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "mlp_dataset"
    / "mlp_w1024_ov50_motor2_vibration_7cls_psd_bands_v2"
)

RUNS_ROOT   = PROJECT_ROOT / "outputs" / "mlp_baseline"
RESULTS_DB  = RUNS_ROOT / "results_db.csv"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 128
EPOCHS     = 100
LR         = 5e-4
SEED       = 42
HIERARCHICAL_L1_ONLY = False
HIER_L2_ONLY = False
HIER_TARGET_CLASS = "bearing pump"

FEATURE_SELECTION_K = 50

CNN_COMPARISON_CLASSES = [
    "bearing pump",
    "broken rotor bar",
    "healthy",
    "impeller",
]

# hiperparâmetros do modelo — altere aqui para cada experimento
MODEL_CFG = {
    "layers": [128, 64],
    "dropout": [0.45, 0.35],
    "input_dropout": 0.25,
    "batch_norm": True,
    "lr": LR,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
}


# =========================================================
# VERSIONING — pasta mlp_vN automática
# =========================================================
def next_run_dir() -> tuple[Path, str]:
    existing = sorted(RUNS_ROOT.glob("mlp_v*"))
    idx = len(existing) + 1
    name = f"mlp_v{idx}"
    path = RUNS_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path, name


# =========================================================
# REPRO
# =========================================================
def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# =========================================================
# LOAD
# =========================================================
def load_npz(path: Path) -> np.ndarray:
    return np.load(path)["X"].astype(np.float32)


def load_labels(path: Path) -> np.ndarray:
    return pd.read_csv(path)["label"].astype(str).values


def load_data():
    print("\n[LOAD] cache:", CACHE_DIR.name)
    X_train = load_npz(CACHE_DIR / "X_train.npz")
    X_val   = load_npz(CACHE_DIR / "X_val.npz")
    X_test  = load_npz(CACHE_DIR / "X_test.npz")

    y_train_raw = load_labels(CACHE_DIR / "y_train.csv")
    y_val_raw   = load_labels(CACHE_DIR / "y_val.csv")
    y_test_raw  = load_labels(CACHE_DIR / "y_test.csv")

    print(f"  X_train {X_train.shape}  X_val {X_val.shape}  X_test {X_test.shape}")
    return X_train, X_val, X_test, y_train_raw, y_val_raw, y_test_raw


# =========================================================
# SPLIT CHECK
# =========================================================
def check_split():
    df = pd.read_csv(CACHE_DIR / "split_intersections.csv")
    print("\n[SPLIT CHECK]")
    print(df.to_string(index=False))
    if df["n_common"].sum() != 0:
        raise RuntimeError("Vazamento detectado nos splits!")


# =========================================================
# ENCODE + SCALE
# =========================================================
def encode_labels(y_train, y_val, y_test, out_dir: Path):
    enc = LabelEncoder()
    # fit em todas as classes para não crashar se val/test tiver classe ausente no treino
    enc.fit(np.concatenate([y_train, y_val, y_test]))
    pd.DataFrame({
        "class_index": range(len(enc.classes_)),
        "class_name":  enc.classes_,
    }).to_csv(out_dir / "label_encoder.csv", index=False)
    return enc.transform(y_train), enc.transform(y_val), enc.transform(y_test), enc




def scale(X_train, X_val, X_test, out_dir: Path):
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val   = np.nan_to_num(X_val,   nan=0.0, posinf=0.0, neginf=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)

    # segura features PSD explosivas antes do scaler
    X_train = np.clip(X_train, -1e6, 1e6)
    X_val   = np.clip(X_val,   -1e6, 1e6)
    X_test  = np.clip(X_test,  -1e6, 1e6)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # segura outlier pós-scaler
    X_train = np.clip(X_train, -8, 8)
    X_val   = np.clip(X_val,   -8, 8)
    X_test  = np.clip(X_test,  -8, 8)

    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return X_train, X_val, X_test


# =========================================================
# FEATURE SELECTION
# =========================================================
def apply_feature_selection(X_train, y_train, X_val, X_test, out_dir: Path, k: int | None):
    if k is None or k <= 0:
        print("\n[FEATURE SELECTION] desativado")
        return X_train, X_val, X_test

    k = min(k, X_train.shape[1])

    selector = SelectKBest(score_func=f_classif, k=k)

    X_train_sel = selector.fit_transform(X_train, y_train)
    X_val_sel   = selector.transform(X_val)
    X_test_sel  = selector.transform(X_test)

    with open(out_dir / "feature_selector.pkl", "wb") as f:
        pickle.dump(selector, f)

    print("\n[FEATURE SELECTION] SelectKBest f_classif")
    print(f"  features: {X_train.shape[1]} -> {X_train_sel.shape[1]}")

    return X_train_sel, X_val_sel, X_test_sel

# =========================================================
# MODEL
# =========================================================
def build_model(n_features: int, n_classes: int, cfg: dict) -> tf.keras.Model:
    layers  = cfg["layers"]
    dropout = cfg["dropout"]
    l2      = cfg.get("l2_reg", 0)
    bn      = cfg.get("batch_norm", True)

    model = Sequential()
    model.add(tf.keras.Input(shape=(n_features,)))

    input_dropout = cfg.get("input_dropout", 0.0)

    if input_dropout > 0:
        model.add(Dropout(input_dropout))

    for i, units in enumerate(layers):
        if l2 > 0:
            model.add(Dense(units, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2)))
        else:
            model.add(Dense(units, activation="relu"))
        if bn:
            model.add(BatchNormalization())
        model.add(Dropout(dropout[i] if i < len(dropout) else 0.2))

    model.add(Dense(n_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=cfg["lr"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# =========================================================
# LIVE PLOT CALLBACK
# =========================================================
class LivePlotCallback(Callback):
    """Salva curvas de loss/accuracy ao final de cada época."""

    def __init__(self, out_dir: Path, run_name: str):
        super().__init__()
        self.out_dir  = out_dir
        self.run_name = run_name
        self._history: dict[str, list] = {
            "loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []
        }

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k in self._history:
            self._history[k].append(logs.get(k, float("nan")))
        self._save_plot(epoch + 1)

    def _save_plot(self, epoch: int):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"{self.run_name}  —  época {epoch}", fontsize=11)

        epochs_range = range(1, epoch + 1)

        axes[0].plot(epochs_range, self._history["loss"],     label="train")
        axes[0].plot(epochs_range, self._history["val_loss"], label="val")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Época")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs_range, self._history["accuracy"],     label="train")
        axes[1].plot(epochs_range, self._history["val_accuracy"], label="val")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Época")
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(self.out_dir / "training_curves_live.png", dpi=100)
        plt.close(fig)


# =========================================================
# PLOTS FINAIS
# =========================================================
def plot_final_curves(hist, out_dir: Path, run_name: str):
    epochs = range(1, len(hist.history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{run_name}  —  curvas de treinamento", fontsize=12)

    axes[0].plot(epochs, hist.history["loss"],     label="train", linewidth=2)
    axes[0].plot(epochs, hist.history["val_loss"], label="val",   linewidth=2, linestyle="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Época")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, hist.history["accuracy"],     label="train", linewidth=2)
    axes[1].plot(epochs, hist.history["val_accuracy"], label="val",   linewidth=2, linestyle="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Época")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "training_curves_final.png", dpi=120)
    plt.close(fig)


def save_cm_count_and_norm_separate(cm, classes, out_dir: Path, run_name: str, prefix: str):
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        out_dir / f"{prefix}_count.csv"
    )
    pd.DataFrame(cm_norm, index=classes, columns=classes).to_csv(
        out_dir / f"{prefix}_normalized.csv"
    )

    # CONTAGEM
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f"{run_name} — {prefix} — Contagem")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    fig.savefig(out_dir / f"{prefix}_count.png", dpi=140)
    plt.close(fig)

    # NORMALIZADA
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        linewidths=0.5,
        vmin=0,
        vmax=1,
    )
    ax.set_title(f"{run_name} — {prefix} — Normalizada")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    fig.savefig(out_dir / f"{prefix}_normalized.png", dpi=140)
    plt.close(fig)

    print(f"\n[MATRIX SEPARADA] {prefix}")
    print(f"  salvo: {prefix}_count.png")
    print(f"  salvo: {prefix}_normalized.png")

def plot_confusion_matrix(y_true, y_pred, classes, out_dir: Path, run_name: str):
    cm = confusion_matrix(y_true, y_pred)

    save_cm_count_and_norm_separate(
        cm=cm,
        classes=classes,
        out_dir=out_dir,
        run_name=run_name,
        prefix="confusion_matrix",
    )


def evaluate_by_group(
    y_true_window,
    y_pred_window,
    enc,
    out_dir: Path,
    run_name: str,
    meta_mask=None,
):
    meta_test_path = CACHE_DIR / "meta_test.csv"

    if not meta_test_path.exists():
        print("\n[WARN] meta_test.csv não encontrado. Pulando avaliação por grupo.")
        return None

    meta = pd.read_csv(meta_test_path).reset_index(drop=True)

    if meta_mask is not None:
        meta = meta.loc[meta_mask].reset_index(drop=True)

    if len(meta) != len(y_true_window):
        raise ValueError(
            f"meta_test.csv tem {len(meta)} linhas, mas y_test tem {len(y_true_window)}."
        )

    group_col = "grupo_condicao"
    if group_col not in meta.columns:
        raise KeyError(f"Coluna '{group_col}' não existe em meta_test.csv")

    df_pred = meta[[group_col]].copy()
    df_pred["y_true"] = y_true_window
    df_pred["y_pred"] = y_pred_window

    grouped = df_pred.groupby(group_col).agg({
        "y_true": lambda x: x.value_counts().idxmax(),
        "y_pred": lambda x: x.value_counts().idxmax(),
    }).reset_index()

    y_true_group = grouped["y_true"].to_numpy()
    y_pred_group = grouped["y_pred"].to_numpy()

    group_acc = accuracy_score(y_true_group, y_pred_group)
    group_f1 = f1_score(y_true_group, y_pred_group, average="macro", zero_division=0)

    classes = enc.classes_

    report_df = pd.DataFrame(
        classification_report(
            y_true_group,
            y_pred_group,
            target_names=classes,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    report_df.to_csv(out_dir / "classification_report_group.csv", index=True)

    grouped["y_true_label"] = enc.inverse_transform(y_true_group)
    grouped["y_pred_label"] = enc.inverse_transform(y_pred_group)
    grouped.to_csv(out_dir / "predictions_group.csv", index=False)

    cm = confusion_matrix(y_true_group, y_pred_group)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{run_name} — confusion matrix por grupo", fontsize=12)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=axes[0],
        linewidths=0.5,
    )
    axes[0].set_title("Contagem")
    axes[0].set_xlabel("Predito")
    axes[0].set_ylabel("Real")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].tick_params(axis="y", rotation=0)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=axes[1],
        linewidths=0.5,
    )
    axes[1].set_title("Normalizada por grupo")
    axes[1].set_xlabel("Predito")
    axes[1].set_ylabel("Real")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_group.png", dpi=120)
    plt.close(fig)

    print("\n[GROUP EVAL]")
    print(f"  group_accuracy={group_acc:.4f}")
    print(f"  group_macro_f1={group_f1:.4f}")
    print(f"  grupos avaliados={len(grouped)}")

    return {
        "group_accuracy": group_acc,
        "group_macro_f1": group_f1,
        "n_test_groups": len(grouped),
    }

def plot_per_class_f1(report_df: pd.DataFrame, out_dir: Path, run_name: str):
    classes = [c for c in report_df.index if c not in ("accuracy", "macro avg", "weighted avg")]
    f1s     = report_df.loc[classes, "f1-score"].values

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(classes, f1s, color="steelblue", edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("F1-score")
    ax.set_title(f"{run_name}  —  F1 por classe (test)")
    ax.axvline(x=report_df.loc["macro avg", "f1-score"], color="red",
               linestyle="--", linewidth=1.2, label="macro avg")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "f1_per_class.png", dpi=120)
    plt.close(fig)


def plot_results_db(db_path: Path):
    """Gráfico comparativo de todas as runs acumuladas."""
    if not db_path.exists():
        return
    db = pd.read_csv(db_path)
    if len(db) < 1:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Comparativo de runs — MLP Baseline", fontsize=12)

    x = range(len(db))
    labels = db["run_name"].tolist()

    axes[0].bar(x, db["test_accuracy"], color="steelblue", edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Test Accuracy")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, db["test_macro_f1"], color="darkorange", edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Test Macro F1")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(db_path.parent / "results_db_chart.png", dpi=120)
    plt.close(fig)


# =========================================================
# RESULTS DATABASE
# =========================================================
def update_results_db(run_name: str, metrics: dict, cfg: dict):
    row = {
        "run_name":        run_name,
        "timestamp":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "test_accuracy":   metrics["accuracy"],
        "test_macro_f1":   metrics["macro_f1"],
        "val_accuracy":    metrics["val_accuracy"],
        "val_macro_f1":    metrics["val_macro_f1"],
        "best_epoch":      metrics["best_epoch"],
        "train_time_sec":  metrics["train_time_sec"],
        "cache_dir":       CACHE_DIR.name,
        **{f"cfg_{k}": str(v) for k, v in cfg.items()},
    }

    if RESULTS_DB.exists():
        db = pd.read_csv(RESULTS_DB)
        db = pd.concat([db, pd.DataFrame([row])], ignore_index=True)
    else:
        db = pd.DataFrame([row])

    db.to_csv(RESULTS_DB, index=False)
    print(f"\n[DB] results_db.csv atualizado — {len(db)} runs registradas")
    return db


class RichEpochLogger(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_start = time.time()
        self.epoch_start = None
        self.total_epochs = self.params.get("epochs", 0)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        elapsed_epoch = time.time() - self.epoch_start
        elapsed_total = time.time() - self.train_start

        done = epoch + 1
        total = self.total_epochs if self.total_epochs else done
        avg_epoch = elapsed_total / done
        eta = avg_epoch * (total - done)

        progress = done / total if total else 1.0
        bar_width = 24
        filled = int(progress * bar_width)
        bar = "━" * filled + "─" * (bar_width - filled)

        console.print(
            f"\n[bold cyan]Época {done}/{total}[/bold cyan] "
            f"[green]{bar}[/green] "
            f"[white]{progress*100:5.1f}%[/white] "
            f"[yellow]{elapsed_epoch:.1f}s/época[/yellow] "
            f"[magenta]ETA {eta/60:.1f} min[/magenta]"
        )

        table = Table(show_header=True, header_style="bold")
        table.add_column("Métrica")
        table.add_column("Train", justify="right")
        table.add_column("Val", justify="right")

        table.add_row(
            "Loss",
            f"{float(logs.get('loss', 0)):.6f}",
            f"{float(logs.get('val_loss', 0)):.6f}",
        )

        table.add_row(
            "Accuracy",
            f"{float(logs.get('accuracy', 0)):.6f}",
            f"{float(logs.get('val_accuracy', 0)):.6f}",
        )

        if "lr" in logs:
            table.add_row(
                "LR",
                f"{float(logs.get('lr', 0)):.6f}",
                "",
            )

        console.print(table)

# =========================================================
# EXPORT TXT SUMMARY
# =========================================================
def export_run_summary_txt(
    out_dir: Path,
    run_name: str,
    model_cfg: dict,
    history,
    class_names,
    f1_values,
    cm,
    cm_group=None,
):
    txt_path = out_dir / "run_summary.txt"

    hist = history.history

    best_val_acc_epoch = int(np.argmax(hist.get("val_accuracy", [0]))) + 1
    best_val_acc = float(np.max(hist.get("val_accuracy", [0])))

    best_val_loss_epoch = int(np.argmin(hist.get("val_loss", [999]))) + 1
    best_val_loss = float(np.min(hist.get("val_loss", [999])))

    final_epoch = len(hist.get("loss", []))

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"RUN SUMMARY — {run_name}\n")
        f.write("=" * 80 + "\n\n")

        f.write("[CONFIG]\n")
        f.write(f"BATCH_SIZE = {BATCH_SIZE}\n")
        f.write(f"EPOCHS = {EPOCHS}\n")
        f.write(f"LR = {LR}\n")
        f.write(f"FEATURE_SELECTION_K = {FEATURE_SELECTION_K}\n")
        f.write(f"MODEL_CFG = {model_cfg}\n")
        f.write(f"HIERARCHICAL_L1_ONLY = {HIERARCHICAL_L1_ONLY}\n")
        f.write(f"HIER_L2_ONLY = {HIER_L2_ONLY}\n\n")

        f.write("[MELHORES ÉPOCAS]\n")
        f.write(f"best val_accuracy: {best_val_acc:.6f} | epoch {best_val_acc_epoch}\n")
        f.write(f"best val_loss:     {best_val_loss:.6f} | epoch {best_val_loss_epoch}\n")
        f.write(f"final epoch:       {final_epoch}\n\n")

        f.write("[ÚLTIMA ÉPOCA]\n")
        for key in ["loss", "accuracy", "val_loss", "val_accuracy", "lr"]:
            if key in hist:
                f.write(f"{key}: {hist[key][-1]:.6f}\n")
        f.write("\n")

        f.write("[F1 POR CLASSE]\n")
        macro_f1 = float(np.mean(f1_values))
        for name, value in zip(class_names, f1_values):
            f.write(f"{name}: {float(value):.6f}\n")
        f.write(f"macro_f1: {macro_f1:.6f}\n\n")

        f.write("[CONFUSION MATRIX]\n")
        f.write("classes: " + ", ".join(map(str, class_names)) + "\n")
        f.write(np.array2string(cm, separator=", "))
        f.write("\n\n")

        if cm_group is not None:
            f.write("[CONFUSION MATRIX POR GRUPO]\n")
            f.write(np.array2string(cm_group, separator=", "))
            f.write("\n\n")

        f.write("[RESUMO COPIÁVEL PARA CHAT]\n")
        f.write(f"{run_name} | K={FEATURE_SELECTION_K} | batch={BATCH_SIZE} | lr={LR}\n")
        f.write(f"macro_f1={macro_f1:.3f}\n")
        for name, value in zip(class_names, f1_values):
            f.write(f"{name}={float(value):.3f} | ")
        f.write("\n")

    print(f"\n[EXPORT TXT] resumo salvo em: {txt_path}")

# =========================================================
# CNN-COMPARABLE FILTERED EVAL
# =========================================================
def evaluate_filtered_classes_for_cnn_comparison(
    y_true,
    y_pred,
    enc,
    out_dir: Path,
    run_name: str,
    keep_classes: list[str],
):
    keep_classes = [c for c in keep_classes if c in enc.classes_]

    if not keep_classes:
        print("\n[CNN FILTER EVAL] nenhuma classe alvo encontrada. Pulando.")
        return None

    keep_ids = enc.transform(keep_classes)
    mask = np.isin(y_true, keep_ids)

    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]

    labels_all = np.arange(len(enc.classes_))
    labels_keep = keep_ids

    report_dict = classification_report(
        y_true_f,
        y_pred_f,
        labels=labels_keep,
        target_names=keep_classes,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(out_dir / "classification_report_cnn4.csv", index=True)

    cm_all = confusion_matrix(
        y_true_f,
        y_pred_f,
        labels=labels_all,
    )

    pd.DataFrame(
        cm_all,
        index=enc.classes_,
        columns=enc.classes_,
    ).to_csv(out_dir / "confusion_matrix_cnn4_all_pred_classes.csv")

    cm_4 = confusion_matrix(
        y_true_f,
        y_pred_f,
        labels=labels_keep,
    )

    row_sums = cm_4.sum(axis=1, keepdims=True)
    cm_4_norm = np.divide(
        cm_4,
        row_sums,
        out=np.zeros_like(cm_4, dtype=float),
        where=row_sums != 0,
    )

    pd.DataFrame(cm_4, index=keep_classes, columns=keep_classes).to_csv(
        out_dir / "confusion_matrix_cnn4.csv"
    )

    pd.DataFrame(cm_4_norm, index=keep_classes, columns=keep_classes).to_csv(
        out_dir / "confusion_matrix_cnn4_normalized.csv"
    )

    save_cm_count_and_norm_separate(
        cm=cm_4,
        classes=keep_classes,
        out_dir=out_dir,
        run_name=run_name,
        prefix="confusion_matrix_group",
    )

    macro_f1 = f1_score(
        y_true_f,
        y_pred_f,
        labels=labels_keep,
        average="macro",
        zero_division=0,
    )

    acc = accuracy_score(y_true_f, y_pred_f)

    print("\n[CNN FILTER EVAL]")
    print("classes:", keep_classes)
    print(f"n_windows={len(y_true_f)}")
    print(f"accuracy={acc:.4f}")
    print(f"macro_f1={macro_f1:.4f}")
    print("salvo: classification_report_cnn4.csv")
    print("salvo: confusion_matrix_cnn4.png")
    print("salvo: confusion_matrix_cnn4_all_pred_classes.csv")

    return {
        "cnn4_accuracy": acc,
        "cnn4_macro_f1": macro_f1,
        "cnn4_n_windows": len(y_true_f),
    }

# =========================================================
# POST EVAL — AGRUPAR LOOSE FOOT
# =========================================================
def evaluate_loose_grouped(
    y_true,
    y_pred,
    enc,
    out_dir: Path,
    run_name: str,
    prefix: str = "mlp_loose_grouped",
):
    def map_label(label: str) -> str:
        if label in ["loose foot motor", "loose foot pump"]:
            return "loose foot"
        return label

    y_true_labels = enc.inverse_transform(y_true)
    y_pred_labels = enc.inverse_transform(y_pred)

    y_true_grouped = np.array([map_label(x) for x in y_true_labels])
    y_pred_grouped = np.array([map_label(x) for x in y_pred_labels])

    classes_grouped = [
        "bearing pump",
        "broken rotor bar",
        "healthy",
        "impeller",
        "loose foot",
    ]

    cm = confusion_matrix(
        y_true_grouped,
        y_pred_grouped,
        labels=classes_grouped,
    )

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    report = classification_report(
        y_true_grouped,
        y_pred_grouped,
        labels=classes_grouped,
        target_names=classes_grouped,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(out_dir / f"{prefix}_classification_report.csv")

    pd.DataFrame(cm, index=classes_grouped, columns=classes_grouped).to_csv(
        out_dir / f"{prefix}_confusion_matrix_count.csv"
    )

    pd.DataFrame(cm_norm, index=classes_grouped, columns=classes_grouped).to_csv(
        out_dir / f"{prefix}_confusion_matrix_normalized.csv"
    )

    # matriz contagem
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes_grouped,
        yticklabels=classes_grouped,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f"{run_name} — Loose foot agrupado — Contagem")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    fig.savefig(out_dir / f"{prefix}_confusion_matrix_count.png", dpi=140)
    plt.close(fig)

    # matriz normalizada
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=classes_grouped,
        yticklabels=classes_grouped,
        ax=ax,
        linewidths=0.5,
        vmin=0,
        vmax=1,
    )
    ax.set_title(f"{run_name} — Loose foot agrupado — Normalizada")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    fig.savefig(out_dir / f"{prefix}_confusion_matrix_normalized.png", dpi=140)
    plt.close(fig)

    acc = accuracy_score(y_true_grouped, y_pred_grouped)
    macro_f1 = f1_score(
        y_true_grouped,
        y_pred_grouped,
        labels=classes_grouped,
        average="macro",
        zero_division=0,
    )

    print("\n[LOOSE GROUPED EVAL]")
    print(f"accuracy={acc:.4f}")
    print(f"macro_f1={macro_f1:.4f}")
    print(f"salvo: {prefix}_confusion_matrix_count.png")
    print(f"salvo: {prefix}_confusion_matrix_normalized.png")
    print(f"salvo: {prefix}_classification_report.csv")

    return {
        "loose_grouped_accuracy": acc,
        "loose_grouped_macro_f1": macro_f1,
    }

# =========================================================
# MAIN
# =========================================================
def main():
    set_seed(SEED)

    out_dir, run_name = next_run_dir()

    console.rule(f"[bold cyan]TRAIN MLP BASELINE — {run_name}[/bold cyan]")
    console.print(f"[green]cache:[/green] {CACHE_DIR.name}")
    console.print(f"[green]saída:[/green] {out_dir}")
    console.print(f"[green]cfg:[/green] {MODEL_CFG}")

    check_split()

    X_train, X_val, X_test, y_train_raw, y_val_raw, y_test_raw = load_data()

    meta_test_mask = None

    if HIERARCHICAL_L1_ONLY:
        NEGATIVE_CLASS = f"non_{HIER_TARGET_CLASS.replace(' ', '_')}"

        y_train_raw = np.where(
            y_train_raw == HIER_TARGET_CLASS,
            HIER_TARGET_CLASS,
            NEGATIVE_CLASS,
        )
        y_val_raw = np.where(
            y_val_raw == HIER_TARGET_CLASS,
            HIER_TARGET_CLASS,
            NEGATIVE_CLASS,
        )
        y_test_raw = np.where(
            y_test_raw == HIER_TARGET_CLASS,
            HIER_TARGET_CLASS,
            NEGATIVE_CLASS,
        )

        print(f"\n[HIER L1] {HIER_TARGET_CLASS} vs {NEGATIVE_CLASS}")
        print("train:", pd.Series(y_train_raw).value_counts().to_dict())
        print("val:  ", pd.Series(y_val_raw).value_counts().to_dict())
        print("test: ", pd.Series(y_test_raw).value_counts().to_dict())

    if HIER_L2_ONLY:
        keep_train = y_train_raw != HIER_TARGET_CLASS
        keep_val   = y_val_raw != HIER_TARGET_CLASS
        keep_test  = y_test_raw != HIER_TARGET_CLASS

        meta_test_mask = keep_test

        X_train = X_train[keep_train]
        X_val   = X_val[keep_val]
        X_test  = X_test[keep_test]

        y_train_raw = y_train_raw[keep_train]
        y_val_raw   = y_val_raw[keep_val]
        y_test_raw  = y_test_raw[keep_test]

        motor_fault_labels = ["broken rotor bar", "loose foot motor"]

        y_train_raw = np.where(np.isin(y_train_raw, motor_fault_labels), "motor_fault", "other")
        y_val_raw   = np.where(np.isin(y_val_raw, motor_fault_labels), "motor_fault", "other")
        y_test_raw  = np.where(np.isin(y_test_raw, motor_fault_labels), "motor_fault", "other")

        print("\n[HIER L2 BIN] motor_fault vs other")
        print("train:", pd.Series(y_train_raw).value_counts().to_dict())
        print("val:  ", pd.Series(y_val_raw).value_counts().to_dict())
        print("test: ", pd.Series(y_test_raw).value_counts().to_dict())


        print(f"\n[HIER L2] removendo {HIER_TARGET_CLASS}")
        print("train:", pd.Series(y_train_raw).value_counts().to_dict())
        print("val:  ", pd.Series(y_val_raw).value_counts().to_dict())
        print("test: ", pd.Series(y_test_raw).value_counts().to_dict())



    y_train, y_val, y_test, enc = encode_labels(
        y_train_raw, y_val_raw, y_test_raw, out_dir
    )

    class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train,
    )

    class_weights = {
        int(cls): float(weight)
        for cls, weight in zip(np.unique(y_train), class_weights_array)
    }

    print("\n[CLASS WEIGHT]")
    for cls_idx, weight in class_weights.items():
        print(f"{enc.classes_[cls_idx]}: {weight:.3f}")

    X_train, X_val, X_test = scale(X_train, X_val, X_test, out_dir)

    X_train, X_val, X_test = apply_feature_selection(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        X_test=X_test,
        out_dir=out_dir,
        k=FEATURE_SELECTION_K,
    )

    model = build_model(X_train.shape[1], len(enc.classes_), MODEL_CFG)
    model.summary()

    live_plot = LivePlotCallback(out_dir, run_name)

    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6, verbose=0),
        live_plot,RichEpochLogger(),
    ]

    print("\n[TRAIN]")
    t0 = time.time()

    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MODEL_CFG["epochs"],
        batch_size=MODEL_CFG["batch_size"],
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0,
    )

    train_time = time.time() - t0
    best_epoch = int(np.argmin(hist.history["val_loss"])) + 1

    # ---- avaliação no val ----
    y_val_pred = np.argmax(model.predict(X_val, batch_size=MODEL_CFG["batch_size"]), axis=1)
    val_acc    = accuracy_score(y_val, y_val_pred)
    val_f1     = f1_score(y_val, y_val_pred, average="macro", zero_division=0)


    
    # ---- avaliação no test ----
    print("\n[TEST]")
    y_pred = np.argmax(
        model.predict(X_test, batch_size=MODEL_CFG["batch_size"]), axis=1
    )
    test_acc = accuracy_score(y_test, y_pred)
    test_f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)

    loose_grouped_metrics = evaluate_loose_grouped(
        y_true=y_test,
        y_pred=y_pred,
        enc=enc,
        out_dir=out_dir,
        run_name=run_name,
    )

    group_metrics = evaluate_by_group(
        y_true_window=y_test,
        y_pred_window=y_pred,
        enc=enc,
        out_dir=out_dir,
        run_name=run_name,
        meta_mask=meta_test_mask,
    )

    cnn4_metrics = evaluate_filtered_classes_for_cnn_comparison(
        y_true=y_test,
        y_pred=y_pred,
        enc=enc,
        out_dir=out_dir,
        run_name=run_name,
        keep_classes=CNN_COMPARISON_CLASSES,
    )

    print(f"\nRESULTADO  {run_name}")
    print(f"  val   accuracy={val_acc:.4f}  macro_f1={val_f1:.4f}")
    print(f"  test  accuracy={test_acc:.4f}  macro_f1={test_f1:.4f}")
    print(f"  best_epoch={best_epoch}  tempo={train_time:.1f}s")

    # ---- classification report ----
    report_dict = classification_report(
        y_test, y_pred,
        target_names=enc.classes_,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(out_dir / "classification_report.csv")

    # ---- plots ----
    plot_final_curves(hist, out_dir, run_name)
    plot_confusion_matrix(y_test, y_pred, enc.classes_, out_dir, run_name)
    plot_per_class_f1(report_df, out_dir, run_name)

    # ---- salva modelo ----
    model.save(out_dir / "mlp_baseline.keras")

    # ---- run summary ----
    metrics = {
        "accuracy":       test_acc,
        "macro_f1":       test_f1,
        "val_accuracy":   val_acc,
        "val_macro_f1":   val_f1,
        "best_epoch":     best_epoch,
        "train_time_sec": train_time,
    }

    if group_metrics is not None:
        metrics.update(group_metrics)

    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=enc.classes_,
        output_dict=True,
        zero_division=0,
    )

    if loose_grouped_metrics is not None:
        metrics.update(loose_grouped_metrics)

    f1_values = np.array([
        report_dict[class_name]["f1-score"]
        for class_name in enc.classes_
    ])

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(enc.classes_)),
    )

    export_run_summary_txt(
        out_dir=out_dir,
        run_name=run_name,
        model_cfg=MODEL_CFG,
        history=hist,
        class_names=enc.classes_,
        f1_values=f1_values,
        cm=cm,
        cm_group=cm_group if "cm_group" in locals() else None,
    )

    # ---- database ----
    db = update_results_db(run_name, metrics, MODEL_CFG)
    plot_results_db(RESULTS_DB)

    print("\n" + "=" * 65)
    print(f"FINALIZADO  —  {run_name}")
    print(f"  test  accuracy={test_acc:.4f}  macro_f1={test_f1:.4f}")
    print(f"  saída: {out_dir}")
    print("=" * 65)

# =========================================================
# EXTRA PLOTS — MATRIZES SEPARADAS
# =========================================================
def plot_confusion_matrix_separate_files(
    y_true,
    y_pred,
    class_names,
    out_dir: Path,
    run_name: str,
    prefix: str,
    keep_classes: list[str] | None = None,
):
    class_names = list(class_names)

    if keep_classes is not None:
        keep_classes = [c for c in keep_classes if c in class_names]
        label_ids = np.array([class_names.index(c) for c in keep_classes])

        mask = np.isin(y_true, label_ids)

        y_true_plot = y_true[mask]
        y_pred_plot = y_pred[mask]

        labels = label_ids
        plot_names = keep_classes
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred

        labels = np.arange(len(class_names))
        plot_names = class_names

    cm = confusion_matrix(
        y_true_plot,
        y_pred_plot,
        labels=labels,
    )

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    pd.DataFrame(cm, index=plot_names, columns=plot_names).to_csv(
        out_dir / f"{prefix}_confusion_matrix_count.csv"
    )

    pd.DataFrame(cm_norm, index=plot_names, columns=plot_names).to_csv(
        out_dir / f"{prefix}_confusion_matrix_normalized.csv"
    )

    # -------- CONTAGEM SEPARADA --------
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=plot_names,
        yticklabels=plot_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f"{run_name} — {prefix} — Matriz de Confusão")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    fig.savefig(out_dir / f"{prefix}_confusion_matrix_count.png", dpi=140)
    plt.close(fig)

    # -------- NORMALIZADA SEPARADA --------
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=plot_names,
        yticklabels=plot_names,
        ax=ax,
        linewidths=0.5,
        vmin=0,
        vmax=1,
    )
    ax.set_title(f"{run_name} — {prefix} — Matriz Normalizada")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    fig.savefig(out_dir / f"{prefix}_confusion_matrix_normalized.png", dpi=140)
    plt.close(fig)

    print(f"\n[EXTRA MATRIX PLOTS] {prefix}")
    print(f"  salvo: {prefix}_confusion_matrix_count.png")
    print(f"  salvo: {prefix}_confusion_matrix_normalized.png")

if __name__ == "__main__":
    main()
