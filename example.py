import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
from wmad.models import WMAD
from wmad.data_utils import inject_unlabeled_labels
from deepod.models.tabular import DeepSAD, RoSAS, FeaWAD, PReNet

warnings.filterwarnings('ignore')
np.random.seed(17)

def generate_aligned_data(n_samples=10000, contamination=0.05):
    """
    Generate complex aligned multi-view data.
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples
    contamination : float
        Proportion of anomalies
    
    Returns:
    --------
    X1, X2, X3 : ndarray
        Three views of the data
    y : ndarray
        Labels (0: normal, 1: anomaly)
    """
    print(f"Generating complex aligned data: {n_samples} samples, {contamination*100:.1f}% contamination...")
    n_anom = int(n_samples * contamination)
    n_norm = int(n_samples * (1-contamination))
    
    # Normal samples - increased variance
    shared_normal = np.random.randn(n_norm, 5) * 1.2
    X1_norm = np.concatenate([shared_normal, np.random.randn(n_norm, 15) * 1.2], axis=1)
    X2_norm = np.concatenate([shared_normal[:, :3], np.random.randn(n_norm, 12) * 1.2], axis=1)
    X3_norm = np.concatenate([shared_normal[:, :2], np.random.randn(n_norm, 8) * 1.2], axis=1)
    
    # Anomaly samples - reduced offset (harder to detect)
    base_anom = np.random.randn(n_anom, 5) * 1.5 + 1.5
    X1_anom = np.concatenate([base_anom, np.random.randn(n_anom, 15) * 1.3], axis=1)
    X2_anom = np.concatenate([base_anom[:, :3], np.random.randn(n_anom, 12) * 1.3], axis=1)
    X3_anom = np.concatenate([base_anom[:, :2], np.random.randn(n_anom, 8) * 1.3], axis=1)
    
    X1 = np.vstack([X1_norm, X1_anom])
    X2 = np.vstack([X2_norm, X2_anom])
    X3 = np.vstack([X3_norm, X3_anom])
    y = np.hstack([np.zeros(n_norm), np.ones(n_anom)])
    
    # Add global noise (more blur)
    X1 += np.random.randn(n_samples, 20) * 0.8
    X2 += np.random.randn(n_samples, 15) * 0.8
    X3 += np.random.randn(n_samples, 10) * 0.8
    
    return X1, X2, X3, y

def main():
    # Generate data
    X1, X2, X3, y = generate_aligned_data()
    
    # Split data: 60% train, 20% val, 20% test
    idx = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=42, stratify=y[train_val_idx])
    
    # Standardize
    scaler1, scaler2, scaler3 = StandardScaler(), StandardScaler(), StandardScaler()
    X1_train = scaler1.fit_transform(X1[train_idx])
    X1_val = scaler1.transform(X1[val_idx])
    X1_test = scaler1.transform(X1[test_idx])
    X2_train = scaler2.fit_transform(X2[train_idx])
    X2_val = scaler2.transform(X2[val_idx])
    X2_test = scaler2.transform(X2[test_idx])
    X3_train = scaler3.fit_transform(X3[train_idx])
    X3_val = scaler3.transform(X3[val_idx])
    X3_test = scaler3.transform(X3[test_idx])
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    
    # Concatenate for baseline methods
    X_train = np.concatenate([X1_train, X2_train, X3_train], axis=1)
    X_val = np.concatenate([X1_val, X2_val, X3_val], axis=1)
    X_test = np.concatenate([X1_test, X2_test, X3_test], axis=1)
    
    # Inject unlabeled samples (90% unlabeled)
    # Label convention: 1=known anomaly, np.nan=unlabeled, 0=known normal
    unlabeled_ratio = 0.9
    y_train_weakly = inject_unlabeled_labels(y_train.copy(), unlabeled_ratio=unlabeled_ratio, 
                                            restore_counts={unlabeled_ratio: 0}, min_anomalies=10, random_state=42)
    
    # Count labeled (not NaN) and unlabeled (NaN)
    n_unlabeled = np.isnan(y_train_weakly).sum()
    n_labeled = len(y_train_weakly) - n_unlabeled
    n_labeled_anomalies = np.nansum(y_train_weakly == 1)  # Known anomalies
    n_labeled_normal = np.nansum(y_train_weakly == 0)  # Known normal
    
    print(f"\nDataset: {len(y)} samples, {y.sum()/len(y)*100:.1f}% contamination")
    print(f"Train: {len(y_train)} samples")
    print(f"  - Labeled: {n_labeled} ({n_labeled/len(y_train)*100:.1f}%)")
    print(f"    * Normal: {int(n_labeled_normal)}")
    print(f"    * Anomaly: {int(n_labeled_anomalies)} ({n_labeled_anomalies/n_labeled*100:.1f}% of labeled)")
    print(f"  - Unlabeled: {n_unlabeled} ({n_unlabeled/len(y_train)*100:.1f}%)")
    print(f"Val: {len(y_val)}, Test: {len(y_test)}")
    
    print("\n" + "="*70)
    print("Training Semi-Supervised Anomaly Detection Methods")
    print("="*70)
    
    results = {}
    EPOCHS = 30  # Reduced for faster training
    
    # 1. WMAD (Multi-view with contrastive learning)
    print("\n[1/5] WMAD (Multi-view, lambda=0.1, eta=0.1)")
    model = WMAD(epochs=EPOCHS, batch_size=64, lr=1e-4, rep_dim=64,
                 hidden_dims_view1='128,64', hidden_dims_view2='128,64', hidden_dims_view3='128,64',
                 device='cpu', beta_=1, k=5, lambda_=0.1, eta1=0.1, eta2=0.1, verbose=-1)
    model.fit(X1_train, X2_train, X3_train, y_train_weakly, X1_val, X2_val, X3_val, y_val)
    model.load_model()
    scores = model.decision_function(X1_test, X2_test, X3_test)
    results['WMAD'] = {
        'AUC-ROC': roc_auc_score(y_test, scores),
        'AUC-PR': average_precision_score(y_test, scores)
    }
    print(f"   AUC-ROC: {results['WMAD']['AUC-ROC']:.4f}, AUC-PR: {results['WMAD']['AUC-PR']:.4f}")
    
    # 2. DeepSAD
    print("\n[2/5] DeepSAD")
    model = DeepSAD(epochs=EPOCHS, batch_size=64, lr=1e-4, hidden_dims='128,64', 
                    rep_dim=32, device='cpu', random_state=42,eta1=0.1,eta2=0.1, verbose=0)
    model.fit(X_train, y_train_weakly, X_val, y_val)
    model.load_model()
    scores = model.decision_function(X_test)
    results['DeepSAD'] = {
        'AUC-ROC': roc_auc_score(y_test, scores),
        'AUC-PR': average_precision_score(y_test, scores)
    }
    print(f"   AUC-ROC: {results['DeepSAD']['AUC-ROC']:.4f}, AUC-PR: {results['DeepSAD']['AUC-PR']:.4f}")
    
    # 3. RoSAS
    print("\n[3/5] RoSAS")
    model = RoSAS(epochs=EPOCHS, batch_size=64, lr=1e-4, hidden_dims='128,64', 
                  device='cpu', random_state=42, verbose=0)
    model.fit(X_train, y_train_weakly, X_val, y_val)
    model.load_model()
    scores = model.decision_function(X_test)
    results['RoSAS'] = {
        'AUC-ROC': roc_auc_score(y_test, scores),
        'AUC-PR': average_precision_score(y_test, scores)
    }
    print(f"   AUC-ROC: {results['RoSAS']['AUC-ROC']:.4f}, AUC-PR: {results['RoSAS']['AUC-PR']:.4f}")
    
    # 4. FeaWAD
    print("\n[4/5] FeaWAD")
    model = FeaWAD(epochs=EPOCHS, batch_size=64, lr=1e-4, hidden_dims='128,64', 
                   rep_dim=32, device='cpu', random_state=42, verbose=0)
    model.fit(X_train, y_train_weakly, X_val, y_val)
    model.load_model()
    scores = model.decision_function(X_test)
    results['FeaWAD'] = {
        'AUC-ROC': roc_auc_score(y_test, scores),
        'AUC-PR': average_precision_score(y_test, scores)
    }
    print(f"   AUC-ROC: {results['FeaWAD']['AUC-ROC']:.4f}, AUC-PR: {results['FeaWAD']['AUC-PR']:.4f}")
    
    # 5. PReNet
    print("\n[5/5] PReNet")
    model = PReNet(epochs=EPOCHS, batch_size=64, lr=1e-4, hidden_dims='128,64', 
                   rep_dim=32, device='cpu', random_state=42, verbose=0)
    model.fit(X_train, y_train_weakly, X_val, y_val)
    model.load_model()
    scores = model.decision_function(X_test)
    results['PReNet'] = {
        'AUC-ROC': roc_auc_score(y_test, scores),
        'AUC-PR': average_precision_score(y_test, scores)
    }
    print(f"   AUC-ROC: {results['PReNet']['AUC-ROC']:.4f}, AUC-PR: {results['PReNet']['AUC-PR']:.4f}")
    
    # Results summary
    sorted_results = sorted(results.items(), key=lambda x: x[1]['AUC-PR'], reverse=True)
    
    print("\n" + "="*70)
    print("Results Summary")
    print("="*70)
    print(f"{'Method':<15} {'AUC-ROC':>10} {'AUC-PR':>10} {'Rank':>8}")
    print("-"*70)
    for rank, (name, res) in enumerate(sorted_results, 1):
        marker = " ***" if name == 'WMAD' else ""
        print(f"{name:<15} {res['AUC-ROC']:>10.4f} {res['AUC-PR']:>10.4f} {rank:>8}{marker}")
    print("="*70)
    
    # Analysis
    best_method = sorted_results[0]
    wmad_result = results['WMAD']
    wmad_rank = next(i for i, (name, _) in enumerate(sorted_results, 1) if name == 'WMAD')
    
    print(f"\nBest method: {best_method[0]} (AUC-PR: {best_method[1]['AUC-PR']:.4f})")
    print(f"WMAD rank: {wmad_rank}/5")
    
    if best_method[0] == 'WMAD':
        second_best = sorted_results[1]
        improvement = (wmad_result['AUC-PR'] - second_best[1]['AUC-PR']) / second_best[1]['AUC-PR'] * 100
        print(f"WMAD outperforms {second_best[0]} by {improvement:.1f}%")

if __name__ == "__main__":
    main()

