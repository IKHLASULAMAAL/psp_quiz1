"""
====================================================================
MEDICATION ADHERENCE CLASSIFICATION - 5 METODE MACHINE LEARNING
====================================================================
Struktur folder (jalankan dari dalam folder datasetkecil/):

  datasetkecil/
  ├── main.py                   ← file ini
  └── medication_adherence.csv  ← dataset

Metode:
  1. K-Nearest Neighbors (KNN)
  2. Naive Bayes (Gaussian)
  3. Backpropagation Neural Network (BPNN / MLPClassifier)
  4. Support Vector Machine (SVM)
  5. Extreme Learning Machine (ELM)

Pipeline:
  Load CSV → Feature Engineering → Preprocessing (Encode + Scale + PCA)
  → Train/Val/Test split → Train 5 Model → Evaluasi → Visualisasi

====================================================================
Cara pakai:
  pip install scikit-learn numpy pandas matplotlib seaborn
  cd datasetkecil
  python main.py
====================================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                   # agar bisa simpan grafik tanpa GUI
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, roc_auc_score,
    ConfusionMatrixDisplay
)

warnings.filterwarnings('ignore')

# ============================================================
#  KONFIGURASI — PATH OTOMATIS (tidak perlu ubah apapun)
# ============================================================
# Direktori tempat main.py berada → datasetkecil/
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_PATH    = os.path.join(BASE_DIR, "medication_adherence.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output_plots")

TARGET_COL     = "future_non_adherence"
TEST_SIZE      = 0.15
VAL_SIZE       = 0.15
USE_PCA        = True
PCA_COMPONENTS = 15
RANDOM_STATE   = 42

# ============================================================
#  1. LOAD & FEATURE ENGINEERING
# ============================================================
def load_and_engineer(csv_path: str) -> pd.DataFrame:
    """
    Load CSV, parse fitur kompleks (blood_pressure, timestamp),
    dan buat fitur turunan baru.
    """
    df = pd.read_csv(csv_path)
    print(f"  File     : {csv_path}")
    print(f"  Dataset  : {df.shape[0]} baris × {df.shape[1]} kolom")

    # --- blood_pressure: "139/81" → dua kolom numerik ---
    bp = df['blood_pressure'].str.split('/', expand=True).astype(float)
    df['bp_systolic']    = bp[0]
    df['bp_diastolic']   = bp[1]
    df['pulse_pressure'] = bp[0] - bp[1]

    # --- timestamp → fitur waktu ---
    df['timestamp']   = pd.to_datetime(df['timestamp'])
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek   # 0=Senin … 6=Minggu
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    df['is_night']    = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] < 6)).astype(int)

    # --- BMI category ---
    df['bmi_category'] = pd.cut(
        df['BMI'], bins=[0, 18.5, 25, 30, 100],
        labels=[0, 1, 2, 3]
    ).astype(float)

    # --- reminder_response_time: isi NaN dengan median ---
    df['has_reminder_response']  = df['reminder_response_time'].notna().astype(int)
    df['reminder_response_time'] = df['reminder_response_time'].fillna(
        df['reminder_response_time'].median()
    )

    # --- missed_reason: isi NaN → 'none', tandai keberadaannya ---
    df['missed_reason_known'] = df['missed_reason'].notna().astype(int)
    df['missed_reason']       = df['missed_reason'].fillna('none')

    # --- Hapus kolom ID & kolom mentah yang sudah di-parse ---
    df = df.drop(columns=['patient_id', 'event_id', 'timestamp', 'blood_pressure'])

    return df


# ============================================================
#  2. PREPROCESSING
# ============================================================
def preprocess(df: pd.DataFrame, target: str):
    """Label-encode semua kolom kategorikal, kembalikan X (array) dan y."""
    df = df.copy()

    cat_cols = ['gender', 'chronic_condition', 'medication_type',
                'device_id', 'location', 'missed_reason']

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=[target])
    y = df[target]

    print(f"\n  Fitur    : {X.shape[1]} kolom")
    print(f"  Target   : {target}")
    print(f"  Kelas    : {dict(y.value_counts().sort_index())}")

    return X.values.astype(np.float32), y.values


# ============================================================
#  ELM — Implementasi Custom
# ============================================================
class ELM:
    """
    Extreme Learning Machine.
    Bobot input (W, b) di-random sekali dan tidak diperbarui.
    Bobot output (beta) dihitung via pseudoinverse: β = H⁺ · T
    """
    def __init__(self, n_hidden=500, activation='relu', random_state=42):
        self.n_hidden     = n_hidden
        self.activation   = activation
        self.random_state = random_state
        self.W = self.b = self.beta = self.classes_ = None

    def _activate(self, X):
        H = X.astype(np.float32) @ self.W + self.b
        if self.activation == 'relu':
            return np.maximum(0, H)
        elif self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(H, -500, 500)))
        return np.tanh(H)

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.classes_ = np.unique(y)
        n_cls = len(self.classes_)
        self.W = rng.randn(X.shape[1], self.n_hidden).astype(np.float32)
        self.b = rng.randn(self.n_hidden).astype(np.float32)
        idx_map = {c: i for i, c in enumerate(self.classes_)}
        T = np.eye(n_cls)[[idx_map[yi] for yi in y]]
        H = self._activate(X)
        self.beta = np.linalg.pinv(H) @ T
        return self

    def predict(self, X):
        return self.classes_[np.argmax(self._activate(X) @ self.beta, axis=1)]

    def predict_proba(self, X):
        scores = self._activate(X) @ self.beta
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# ============================================================
#  EVALUASI
# ============================================================
def evaluate(name, model, X_test, y_test, fit_time,
             predict_fn=None, proba_fn=None):
    t0     = time.time()
    y_pred = predict_fn(X_test) if predict_fn else model.predict(X_test)
    infer  = time.time() - t0

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='macro',  zero_division=0)
    f1b  = f1_score(y_test, y_pred, pos_label=1,      zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    try:
        if proba_fn:
            prob = proba_fn(X_test)[:, 1]
        elif hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            prob = model.decision_function(X_test)
        else:
            prob = None
        auc = roc_auc_score(y_test, prob) if prob is not None else float('nan')
    except Exception:
        auc = float('nan')

    print(f"\n{'='*62}")
    print(f"  HASIL : {name}")
    print(f"{'='*62}")
    print(f"  Akurasi              : {acc*100:.2f}%")
    print(f"  F1-Score Macro       : {f1*100:.2f}%")
    print(f"  F1 Non-Adherent (1)  : {f1b*100:.2f}%")
    print(f"  ROC-AUC              : {auc:.4f}")
    print(f"  Waktu Latih          : {fit_time:.4f} detik")
    print(f"  Waktu Prediksi       : {infer:.4f} detik")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=['Adherent(0)', 'NonAdherent(1)'], zero_division=0))

    return dict(name=name, accuracy=acc, f1=f1, f1b=f1b,
                auc=auc, fit_time=fit_time, infer_time=infer,
                cm=cm, y_pred=y_pred)


# ============================================================
#  VISUALISASI
# ============================================================
def plot_eda(df_raw, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # 1. Distribusi target
    counts = df_raw[TARGET_COL].value_counts().sort_index()
    axes[0].bar(['Adherent (0)', 'Non-Adherent (1)'],
                counts.values, color=['#4CAF50', '#F44336'])
    axes[0].set_title('Distribusi Target', fontweight='bold')
    axes[0].set_ylabel('Jumlah')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')

    # 2. Distribusi usia per kelas
    for lbl, clr in [(0, '#4CAF50'), (1, '#F44336')]:
        axes[1].hist(df_raw[df_raw[TARGET_COL] == lbl]['age'],
                     bins=20, alpha=0.6, color=clr,
                     label='Adherent' if lbl == 0 else 'Non-Adherent')
    axes[1].set_title('Distribusi Usia per Target', fontweight='bold')
    axes[1].set_xlabel('Usia')
    axes[1].legend()

    # 3. % Non-Adherent per stress level
    stress = df_raw.groupby('stress_level')[TARGET_COL].mean() * 100
    axes[2].bar(stress.index, stress.values, color='#9C27B0')
    axes[2].set_title('% Non-Adherent per Stress Level', fontweight='bold')
    axes[2].set_xlabel('Stress Level')
    axes[2].set_ylabel('% Non-Adherent')

    plt.suptitle('EDA — Medication Adherence Dataset',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → EDA plot          : {save_path}")


def plot_cm(cm, title, save_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Adherent', 'Non-Adherent']
    ).plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_comparison(results, save_path):
    names  = [r['name']       for r in results]
    accs   = [r['accuracy']*100 for r in results]
    f1bs   = [r['f1b']*100   for r in results]
    aucs   = [r['auc']        for r in results]
    times  = [r['fit_time']   for r in results]
    colors = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#F44336']

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    def bar(ax, vals, title, fmt="{:.1f}%", ylim=(0, 110)):
        ax.bar(names, vals, color=colors, edgecolor='white')
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_ylim(*ylim)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
        for i, v in enumerate(vals):
            ax.text(i, v + ylim[1]*0.01, fmt.format(v), ha='center', fontsize=8.5)

    bar(axes[0], accs,  'Akurasi (%)')
    bar(axes[1], f1bs,  'F1 Non-Adherent (%)')
    bar(axes[2], aucs,  'ROC-AUC', fmt="{:.3f}", ylim=(0, 1.15))
    bar(axes[3], times, 'Waktu Latih (detik)', fmt="{:.3f}s",
        ylim=(0, max(times) * 1.3 + 0.01))

    plt.suptitle('Perbandingan Model — Medication Adherence',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Grafik perbandingan: {save_path}")


# ============================================================
#  MAIN PIPELINE
# ============================================================
def main():
    print("=" * 62)
    print("  MEDICATION ADHERENCE — 5 METODE MACHINE LEARNING")
    print("=" * 62)
    print(f"\n  Base dir : {BASE_DIR}")
    print(f"  CSV      : {CSV_PATH}")

    # Cek file CSV ada
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(
            f"\n[ERROR] File tidak ditemukan: {CSV_PATH}\n"
            f"Pastikan 'medication_adherence.csv' ada di folder yang sama dengan main.py\n"
            f"Folder saat ini: {BASE_DIR}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"  Output   : {OUTPUT_DIR}\n")

    # ----------------------------------------------------------
    # STEP 1 — Load & Feature Engineering
    # ----------------------------------------------------------
    print("[1] Load & Feature Engineering...")
    df_raw = pd.read_csv(CSV_PATH)
    df     = load_and_engineer(CSV_PATH)
    plot_eda(df_raw, os.path.join(OUTPUT_DIR, "eda.png"))

    # ----------------------------------------------------------
    # STEP 2 — Preprocessing & Split
    # ----------------------------------------------------------
    print("\n[2] Preprocessing & Split...")
    X, y = preprocess(df, TARGET_COL)

    # Stratified split: 70% train | 15% val | 15% test
    X_tv,  X_test,  y_tv,  y_test  = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE, stratify=y_tv)

    print(f"\n  Train      : {len(X_train):>5} sampel  (non-adh={y_train.sum()})")
    print(f"  Validation : {len(X_val):>5} sampel  (non-adh={y_val.sum()})")
    print(f"  Test       : {len(X_test):>5} sampel  (non-adh={y_test.sum()})")

    # Gabung train + val untuk fit final
    X_fit = np.vstack([X_train, X_val])
    y_fit = np.concatenate([y_train, y_val])

    # Scaling
    scaler    = StandardScaler()
    X_fit_sc  = scaler.fit_transform(X_fit)
    X_test_sc = scaler.transform(X_test)

    # PCA
    if USE_PCA:
        n_comp     = min(PCA_COMPONENTS, X_fit_sc.shape[0] - 1, X_fit_sc.shape[1])
        pca        = PCA(n_components=n_comp, random_state=RANDOM_STATE)
        X_fit_pca  = pca.fit_transform(X_fit_sc)
        X_test_pca = pca.transform(X_test_sc)
        print(f"\n  PCA: {n_comp} komponen → {pca.explained_variance_ratio_.sum()*100:.1f}% varians")
    else:
        X_fit_pca, X_test_pca = X_fit_sc, X_test_sc

    results = []

    # ----------------------------------------------------------
    # MODEL 1 — KNN
    # ----------------------------------------------------------
    print("\n" + "─"*62)
    print("[MODEL 1] K-Nearest Neighbors (KNN)")
    t0  = time.time()
    knn = KNeighborsClassifier(n_neighbors=7, metric='euclidean',
                               weights='distance', n_jobs=-1)
    knn.fit(X_fit_pca, y_fit)
    results.append(evaluate("KNN", knn, X_test_pca, y_test,
                             fit_time=time.time()-t0))

    # ----------------------------------------------------------
    # MODEL 2 — Naive Bayes
    # ----------------------------------------------------------
    print("\n" + "─"*62)
    print("[MODEL 2] Gaussian Naive Bayes")
    t0 = time.time()
    nb = GaussianNB(var_smoothing=1e-9)
    nb.fit(X_fit_pca, y_fit)
    results.append(evaluate("Naive Bayes", nb, X_test_pca, y_test,
                             fit_time=time.time()-t0))

    # ----------------------------------------------------------
    # MODEL 3 — BPNN (MLP)
    # ----------------------------------------------------------
    print("\n" + "─"*62)
    print("[MODEL 3] Backpropagation Neural Network (BPNN/MLP)")
    t0   = time.time()
    bpnn = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=500,
        batch_size=64,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=RANDOM_STATE
    )
    bpnn.fit(X_fit_pca, y_fit)
    results.append(evaluate("BPNN", bpnn, X_test_pca, y_test,
                             fit_time=time.time()-t0))

    # ----------------------------------------------------------
    # MODEL 4 — SVM
    # ----------------------------------------------------------
    print("\n" + "─"*62)
    print("[MODEL 4] Support Vector Machine (SVM)")
    t0  = time.time()
    svm = SVC(
        C=10, kernel='rbf', gamma='scale',
        class_weight='balanced',   # tangani class imbalance (75:25)
        probability=True,          # agar predict_proba tersedia untuk AUC
        decision_function_shape='ovr',
        random_state=RANDOM_STATE
    )
    svm.fit(X_fit_pca, y_fit)
    results.append(evaluate("SVM", svm, X_test_pca, y_test,
                             fit_time=time.time()-t0))

    # ----------------------------------------------------------
    # MODEL 5 — ELM
    # ----------------------------------------------------------
    print("\n" + "─"*62)
    print("[MODEL 5] Extreme Learning Machine (ELM)")
    t0  = time.time()
    elm = ELM(n_hidden=1000, activation='relu', random_state=RANDOM_STATE)
    elm.fit(X_fit_pca, y_fit)
    results.append(evaluate("ELM", elm, X_test_pca, y_test,
                             fit_time=time.time()-t0,
                             predict_fn=elm.predict,
                             proba_fn=elm.predict_proba))

    # ----------------------------------------------------------
    # VISUALISASI
    # ----------------------------------------------------------
    print("\n[3] Menyimpan visualisasi...")
    for r in results:
        path = os.path.join(OUTPUT_DIR, f"cm_{r['name'].replace(' ','_')}.png")
        plot_cm(r['cm'], title=f"Confusion Matrix — {r['name']}", save_path=path)
        print(f"  → CM {r['name']:<12}: {path}")

    plot_comparison(results, os.path.join(OUTPUT_DIR, "comparison.png"))

    # ----------------------------------------------------------
    # RINGKASAN AKHIR
    # ----------------------------------------------------------
    print("\n" + "="*72)
    print("  RINGKASAN PERBANDINGAN MODEL — MEDICATION ADHERENCE")
    print("="*72)
    print(f"{'Model':<14} {'Akurasi':>9} {'F1-Macro':>10} {'F1-NonAdh':>11} {'ROC-AUC':>9} {'Waktu':>9}")
    print("-"*72)
    for r in results:
        print(f"{r['name']:<14}"
              f" {r['accuracy']*100:>8.2f}%"
              f" {r['f1']*100:>9.2f}%"
              f" {r['f1b']*100:>10.2f}%"
              f" {r['auc']:>9.4f}"
              f" {r['fit_time']:>8.4f}s")
    print("="*72)

    best_acc = max(results, key=lambda r: r['accuracy'])
    best_auc = max(results, key=lambda r: r['auc'])
    print(f"\n  🏆 Akurasi tertinggi : {best_acc['name']} ({best_acc['accuracy']*100:.2f}%)")
    print(f"  🏆 AUC tertinggi     : {best_auc['name']} ({best_auc['auc']:.4f})")
    print(f"\n  Semua grafik tersimpan di → {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()