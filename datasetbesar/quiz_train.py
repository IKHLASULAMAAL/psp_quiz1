"""
====================================================================
TOMATO DISEASE CLASSIFICATION - IMPLEMENTASI 5 METODE MACHINE LEARNING
====================================================================
Struktur folder dataset (letakkan file ini di root project QUIZ1):

  QUIZ1/
  ├── quiz_train.py          ← file ini
  ├── test/
  │     ├── Tomato_Bacterial_spot/
  │     ├── Tomato_healthy/
  │     └── ...
  ├── train/
  │     ├── Tomato_Bacterial_spot/
  │     ├── Tomato_healthy/
  │     └── ...
  └── validation/
        ├── Tomato_Bacterial_spot/
        ├── Tomato_healthy/
        └── ...

Metode:
  1. K-Nearest Neighbors (KNN)
  2. Naive Bayes (Gaussian)
  3. Backpropagation Neural Network (BPNN / MLPClassifier)
  4. Support Vector Machine (SVM)
  5. Extreme Learning Machine (ELM)
====================================================================
Cara pakai:
  pip install scikit-learn numpy pillow matplotlib seaborn tqdm
  python quiz_train.py
====================================================================
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)

# ============================================================
#  KONFIGURASI
# ============================================================
# Dataset berada di folder yang sama dengan script ini (root QUIZ1)
# Struktur: ./train/  ./test/  ./validation/
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))  # folder tempat script ini
IMG_SIZE       = (64, 64)      # resize semua gambar ke 64x64
USE_PCA        = True          # PCA untuk reduksi dimensi
PCA_COMPONENTS = 100           # jumlah komponen PCA
RANDOM_STATE   = 42

# ============================================================
#  FUNGSI LOAD DATASET
# ============================================================
def load_split(base_dir: str, split: str) -> tuple:
    """
    Memuat gambar dari folder split (train / test / validation).
    Folder split berada LANGSUNG di root project (BASE_DIR):
        BASE_DIR/
          train/   class1/ class2/ ...
          test/    class1/ class2/ ...
          validation/ class1/ class2/ ...
    """
    split_path = os.path.join(base_dir, split)

    if not os.path.isdir(split_path):
        raise FileNotFoundError(
            f"Folder '{split}' tidak ditemukan di: {split_path}\n"
            f"Pastikan script dijalankan dari direktori QUIZ1."
        )

    images, labels = [], []
    classes = sorted([
        d for d in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, d))
    ])
    print(f"\n[{split.upper()}] Kelas ditemukan ({len(classes)}): {classes}")

    for cls in classes:
        cls_path = os.path.join(split_path, cls)
        files = [f for f in os.listdir(cls_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"  - {cls}: {len(files)} gambar")
        for fname in tqdm(files, desc=f"  Loading {cls}", leave=False):
            try:
                img = Image.open(os.path.join(cls_path, fname)).convert("RGB")
                img = img.resize(IMG_SIZE)
                images.append(np.array(img).flatten().astype(np.float32) / 255.0)
                labels.append(cls)
            except Exception as e:
                print(f"  [SKIP] {fname}: {e}")

    return np.array(images), np.array(labels)


# ============================================================
#  EXTREME LEARNING MACHINE (ELM) — Implementasi Custom
# ============================================================
class ELM:
    """
    Single-Hidden Layer Feedforward Network (SLFN) / ELM.
    Bobot input (W, b) di-random dan di-fix.
    Bobot output (beta) dihitung via Moore-Penrose pseudoinverse.
    """
    def __init__(self, n_hidden: int = 1000, activation: str = 'relu',
                 random_state: int = 42):
        self.n_hidden    = n_hidden
        self.activation  = activation
        self.random_state = random_state
        self.W    = None  # bobot input  (n_features, n_hidden)
        self.b    = None  # bias hidden  (n_hidden,)
        self.beta = None  # bobot output (n_hidden, n_classes)
        self.le   = LabelEncoder()

    def _activate(self, X: np.ndarray) -> np.ndarray:
        H = X @ self.W + self.b
        if self.activation == 'relu':
            return np.maximum(0, H)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(H, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(H)
        return H

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        self.W = rng.randn(n_features, self.n_hidden).astype(np.float32)
        self.b = rng.randn(self.n_hidden).astype(np.float32)

        # Encode label ke one-hot
        y_enc = self.le.fit_transform(y)
        T = np.eye(len(self.le.classes_))[y_enc]  # one-hot

        H = self._activate(X)                      # hidden output matrix
        # beta = H^+ * T  (pseudoinverse)
        self.beta = np.linalg.pinv(H) @ T
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        H = self._activate(X)
        scores = H @ self.beta
        idx    = np.argmax(scores, axis=1)
        return self.le.inverse_transform(idx)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))


# ============================================================
#  FUNGSI EVALUASI
# ============================================================
def evaluate(name: str, model, X_test, y_test,
             fit_time: float, predict_fn=None):
    t0 = time.time()
    if predict_fn:
        y_pred = predict_fn(X_test)
    else:
        y_pred = model.predict(X_test)
    infer_time = time.time() - t0

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro', zero_division=0)
    cm  = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"  HASIL: {name}")
    print(f"{'='*60}")
    print(f"  Akurasi     : {acc*100:.2f}%")
    print(f"  F1-Score    : {f1*100:.2f}%")
    print(f"  Waktu Latih : {fit_time:.2f} detik")
    print(f"  Waktu Prediksi: {infer_time:.4f} detik")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "name": name, "accuracy": acc, "f1": f1,
        "fit_time": fit_time, "infer_time": infer_time,
        "cm": cm, "y_pred": y_pred
    }


def plot_confusion_matrix(cm: np.ndarray, classes: list,
                          title: str, save_path: str = None):
    plt.figure(figsize=(max(8, len(classes)), max(6, len(classes)-2)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Label Sebenarnya')
    plt.xlabel('Label Prediksi')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_comparison(results: list, save_path: str = None):
    names = [r['name'] for r in results]
    accs  = [r['accuracy']*100 for r in results]
    f1s   = [r['f1']*100 for r in results]
    times = [r['fit_time'] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#2196F3','#4CAF50','#FF9800','#9C27B0','#F44336']

    axes[0].bar(names, accs, color=colors)
    axes[0].set_title('Akurasi (%)', fontweight='bold')
    axes[0].set_ylim(0, 105)
    for i, v in enumerate(accs):
        axes[0].text(i, v+0.5, f"{v:.1f}%", ha='center', fontsize=9)

    axes[1].bar(names, f1s, color=colors)
    axes[1].set_title('F1-Score Macro (%)', fontweight='bold')
    axes[1].set_ylim(0, 105)
    for i, v in enumerate(f1s):
        axes[1].text(i, v+0.5, f"{v:.1f}%", ha='center', fontsize=9)

    axes[2].bar(names, times, color=colors)
    axes[2].set_title('Waktu Pelatihan (detik)', fontweight='bold')
    for i, v in enumerate(times):
        axes[2].text(i, v+0.1, f"{v:.1f}s", ha='center', fontsize=9)

    for ax in axes:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha='right')

    plt.suptitle('Perbandingan Model — Tomato Disease Classification',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
#  MAIN PIPELINE
# ============================================================
def main():
    print("=" * 60)
    print("  TOMATO DISEASE CLASSIFICATION — 5 METODE ML")
    print("=" * 60)

    # --- 1. Load Data ---
    print("\n[1] Memuat dataset...")
    print(f"    Base directory : {BASE_DIR}")
    X_train, y_train = load_split(BASE_DIR, "train")
    X_val,   y_val   = load_split(BASE_DIR, "validation")
    X_test,  y_test  = load_split(BASE_DIR, "test")

    # Gabungkan train + validation untuk pelatihan akhir
    X_fit = np.vstack([X_train, X_val])
    y_fit = np.concatenate([y_train, y_val])

    classes = sorted(np.unique(y_test))
    print(f"\nTotal data latih (train+val): {len(X_fit)}")
    print(f"Total data uji             : {len(X_test)}")
    print(f"Jumlah kelas               : {len(classes)}")
    print(f"Dimensi fitur (raw)        : {X_fit.shape[1]}")

    # --- 2. Scaling & PCA ---
    print("\n[2] Preprocessing: StandardScaler + PCA...")
    scaler = StandardScaler()
    X_fit_sc  = scaler.fit_transform(X_fit)
    X_test_sc = scaler.transform(X_test)

    if USE_PCA:
        n_comp = min(PCA_COMPONENTS, X_fit_sc.shape[0], X_fit_sc.shape[1])
        pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
        X_fit_pca  = pca.fit_transform(X_fit_sc)
        X_test_pca = pca.transform(X_test_sc)
        var_ratio  = pca.explained_variance_ratio_.sum()
        print(f"  PCA: {n_comp} komponen, varians tertangkap: {var_ratio*100:.1f}%")
    else:
        X_fit_pca  = X_fit_sc
        X_test_pca = X_test_sc

    results = []

    # --------------------------------------------------------
    #  MODEL 1: KNN
    # --------------------------------------------------------
    print("\n[MODEL 1] K-Nearest Neighbors (KNN)")
    t0 = time.time()
    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric='euclidean',
        weights='distance',
        n_jobs=-1
    )
    knn.fit(X_fit_pca, y_fit)
    results.append(evaluate("KNN", knn, X_test_pca, y_test,
                             fit_time=time.time()-t0))

    # --------------------------------------------------------
    #  MODEL 2: Naive Bayes
    # --------------------------------------------------------
    print("\n[MODEL 2] Gaussian Naive Bayes")
    t0 = time.time()
    nb = GaussianNB(var_smoothing=1e-9)
    nb.fit(X_fit_pca, y_fit)
    results.append(evaluate("Naive Bayes", nb, X_test_pca, y_test,
                             fit_time=time.time()-t0))

    # --------------------------------------------------------
    #  MODEL 3: BPNN (MLP)
    # --------------------------------------------------------
    print("\n[MODEL 3] Backpropagation Neural Network (BPNN/MLP)")
    t0 = time.time()
    bpnn = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=300,
        batch_size=64,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_STATE
    )
    bpnn.fit(X_fit_pca, y_fit)
    results.append(evaluate("BPNN", bpnn, X_test_pca, y_test,
                             fit_time=time.time()-t0))

    # --------------------------------------------------------
    #  MODEL 4: SVM
    # --------------------------------------------------------
    print("\n[MODEL 4] Support Vector Machine (SVM)")
    t0 = time.time()
    svm = SVC(
        C=10,
        kernel='rbf',
        gamma='scale',
        decision_function_shape='ovr',
        random_state=RANDOM_STATE
    )
    svm.fit(X_fit_pca, y_fit)
    results.append(evaluate("SVM", svm, X_test_pca, y_test,
                             fit_time=time.time()-t0))

    # --------------------------------------------------------
    #  MODEL 5: ELM
    # --------------------------------------------------------
    print("\n[MODEL 5] Extreme Learning Machine (ELM)")
    t0 = time.time()
    elm = ELM(n_hidden=2000, activation='relu', random_state=RANDOM_STATE)
    elm.fit(X_fit_pca, y_fit)
    fit_time_elm = time.time() - t0
    results.append(evaluate("ELM", elm, X_test_pca, y_test,
                             fit_time=fit_time_elm,
                             predict_fn=elm.predict))

    # --------------------------------------------------------
    #  VISUALISASI
    # --------------------------------------------------------
    print("\n[3] Menyimpan visualisasi...")
    os.makedirs("output_plots", exist_ok=True)

    # Confusion matrix per model
    for r in results:
        plot_confusion_matrix(
            r['cm'], classes,
            title=f"Confusion Matrix — {r['name']}",
            save_path=f"output_plots/cm_{r['name'].replace(' ','_')}.png"
        )

    # Grafik perbandingan
    plot_comparison(results, save_path="output_plots/comparison.png")

    # --------------------------------------------------------
    #  RINGKASAN AKHIR
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("  RINGKASAN PERBANDINGAN MODEL")
    print("="*60)
    print(f"{'Model':<15} {'Akurasi':>10} {'F1-Score':>10} {'Waktu Latih':>14}")
    print("-"*52)
    for r in results:
        print(f"{r['name']:<15} {r['accuracy']*100:>9.2f}% "
              f"{r['f1']*100:>9.2f}%  {r['fit_time']:>10.2f}s")
    print("="*60)

    best = max(results, key=lambda r: r['accuracy'])
    print(f"\n  🏆 Model terbaik: {best['name']} "
          f"({best['accuracy']*100:.2f}% akurasi)\n")


if __name__ == "__main__":
    main()