# WMAD: Weakly-supervised Multi-view Anomaly Detection

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

WMAD is a deep learning framework for multi-view anomaly detection in weakly-supervised scenarios with limited labeled samples.

## Installation

```bash
git clone https://github.com/GoneZC/WMAD.git
cd WMAD
pip install -r requirements.txt
```

## Quick Start

```python
from wmad.models import WMAD
from wmad.data_utils import inject_unlabeled_labels

# Prepare multi-view data
X1_train, X2_train, X3_train = ...  # Three views
y_train = ...  # Labels: 1=anomaly, 0=normal, np.nan=unlabeled

# Train model
model = WMAD(epochs=30, batch_size=64, lr=1e-4, lambda_=0.1, eta1=0.1, eta2=0.1)
model.fit(X1_train, X2_train, X3_train, y_train, X1_val, X2_val, X3_val, y_val)

# Predict
scores = model.decision_function(X1_test, X2_test, X3_test)
```

See `example.py` for a complete working example with synthetic data and baseline comparisons.

## Acknowledgments

This project is built upon the [DeepOD](https://github.com/xuhongzuo/DeepOD) library.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
