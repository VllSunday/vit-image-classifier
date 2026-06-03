| Experiment | Pretrained | Trainable params | Epochs to best | Test accuracy | Macro F1 |
|---|:---:|---:|---:|---:|---:|
| Linear probe (full data) | ✅ | 2,307 | 3 | 0.9956 | 0.9955 |
| Gradual unfreeze (full data) | ✅ | 28,355,331 | 1 | 0.9933 | 0.9933 |
| Linear probe (small train) | ✅ | 2,307 | 3 | 0.9933 | 0.9933 |
| Gradual unfreeze (small train) | ✅ | 28,355,331 | 3 | 0.9933 | 0.9933 |
| From scratch (no pretraining) | ❌ | 85,800,963 | 17 | 0.6689 | 0.6647 |
