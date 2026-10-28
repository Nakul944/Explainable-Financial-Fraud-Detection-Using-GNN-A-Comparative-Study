# Graph Attention Network (GAT) — Experiment Report

## 1. Objective

This experiment evaluates a Graph Attention Network (GAT) for transaction-level illicit/fraud detection on the Elliptic Bitcoin transaction graph. It is part of the project's comparative study of GCN, GraphSAGE, and GAT.

The aim is to test whether learned attention over neighboring transactions improves fraud detection under the same temporal evaluation setup used by the other models.

## 2. Dataset and preprocessing

- **Dataset:** Elliptic Bitcoin transaction dataset.
- **Task:** Binary classification of known transaction labels.
- **Label mapping:** Legitimate = 0; illicit/fraud = 1; unknown = -1.
- **Unknown labels:** Excluded from supervised train, validation, and test masks.
- **Graph setup:** The preprocessed transaction graph is used for full-graph (transductive) message passing.
- **Temporal split:** Training time steps 1–34, validation time steps 35–41, and test time steps 42–49, restricted to transactions with known labels.
- **Split sizes:** Train = 29,894; validation = 7,829; test = 8,841.

The temporal split evaluates the model on later time steps than those used for training. The graph is used in full-graph message passing; therefore this should be described as a temporal label split within a transductive graph setup, not as a fully isolated unseen-graph evaluation.

## 3. Model architecture

The implementation uses a two-layer Graph Attention Network:

| Setting | Value |
|---|---:|
| GAT layers | 2 |
| First-layer attention heads | 4 |
| Channels per attention head | 16 |
| First-layer output width | 64 |
| Output attention heads | 1 |
| Feature dropout | 0.5 |
| Attention dropout | 0.5 |
| Random seed | 42 |

The first layer uses multi-head attention to aggregate information from neighboring nodes. The output layer produces scores for the two classes.

## 4. Training configuration

| Setting | Value |
|---|---:|
| Maximum epochs | 100 |
| Optimizer | Adam |
| Learning rate | 0.01 |
| Weight decay | 0.0005 |
| Loss | Class-weighted cross-entropy |
| Checkpoint selection | Best validation fraud F1 |
| Best validation fraud F1 | 0.37795 |

The checkpoint selected by validation fraud F1 was restored before the final test evaluation. The test set was not used for checkpoint selection.

## 5. Test-set results

The following metrics are from the recorded test evaluation at the notebook's default class-decision rule.

| Metric | Result |
|---|---:|
| Accuracy | 0.74686 |
| Fraud precision | 0.10931 |
| Fraud recall | 0.62745 |
| Fraud F1-score | 0.18618 |
| PR-AUC (average precision) | 0.30942 |
| ROC-AUC | 0.80733 |

### Classification report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Legitimate (0) | 0.9766 | 0.7526 | 0.8501 | 8,433 |
| Fraud / illicit (1) | 0.1093 | 0.6275 | 0.1862 | 408 |
| Macro average | 0.5430 | 0.6900 | 0.5182 | 8,841 |
| Weighted average | 0.9366 | 0.7469 | 0.8195 | 8,841 |

## 6. Interpretation

The model identifies approximately 62.75% of the fraudulent transactions in the test set, giving it the highest fraud recall among the current recorded GCN, GraphSAGE, and GAT results. However, fraud precision is only 10.93%, which means that most transactions predicted as fraud are actually legitimate.

The fraud F1-score (0.1862) and PR-AUC (0.3094) indicate that this operating point and the current ranking performance are weaker than the recorded GraphSAGE results. The ROC-AUC of 0.8073 shows that the model still has some ability to rank positive cases above negative cases, but ROC-AUC alone can be optimistic on a strongly imbalanced dataset. PR-AUC and class-specific precision/recall are particularly important for this task.

The gap between recall and precision may reflect the interaction of class-weighted training, the default decision threshold, model capacity, and the graph's feature/neighbor patterns. These are possible explanations, not confirmed causes. GAT's attention mechanism does not guarantee that the learned attention weights will correspond to useful fraud signals.

## 7. Comparison with other models

Based on the currently recorded results:

- **GCN:** Fraud precision approximately 0.47, recall approximately 0.43, and F1 approximately 0.45.
- **GraphSAGE:** Fraud precision 0.4421, recall 0.4583, F1 0.4501, PR-AUC 0.4641, ROC-AUC 0.8687.
- **GAT:** Fraud precision 0.1093, recall 0.6275, F1 0.1862, PR-AUC 0.3094, ROC-AUC 0.8073.

GraphSAGE currently has the strongest recorded fraud F1 and PR-AUC among the results available for comparison. GAT prioritizes recall at the current decision rule but incurs many false positives. GCN's PR-AUC and ROC-AUC should be added to the comparison only from its saved evaluation artifacts; they are not reproduced here because exact values were not available in the experiment metadata consulted for this report.

## 8. Limitations and next steps

1. Inspect the saved confusion matrix to quantify false positives and false negatives.
2. If threshold tuning is pursued, select a threshold using validation predictions only. Then evaluate that fixed threshold on the test set once and report the tuned run separately from the original result.
3. Compare validation PR-AUC and validation fraud F1 curves to diagnose training stability and generalization.
4. If running further experiments, vary one factor at a time (for example learning rate, dropout, number of attention heads, or class-weighting strategy) and keep the temporal splits and evaluation protocol fixed.
5. Keep the initial test metrics unchanged in the report for a transparent baseline. Do not choose hyperparameters or thresholds based on test performance.
6. Treat model comparison as provisional until the exact training/evaluation configuration and saved metrics for all models have been checked for consistency.

## 9. Output artifacts

Running `model/03_GAT_Model.ipynb` locally generates the following files under `results/GAT/`:

- `training_loss.png`
- `validation_f1.png`
- `validation_pr_auc.png`
- `confusion_matrix.png`
- `precision_recall_curve.png`
- `roc_curve.png`
- `metrics.json`
- `metrics.csv`

The checkpoint is saved locally as `model/best_gat_model.pt`; experiment metadata is saved as `model/gat_metrics.json`. Checkpoints and the processed graph are intentionally excluded from Git.

## 10. Conclusion

Under the current configuration and default decision rule, GAT provides relatively high fraud recall but low fraud precision, resulting in a fraud F1-score of 0.1862. Its results do not establish GAT as the best-performing model for this dataset. Instead, they show that learned neighbor attention alone is insufficient to guarantee improved fraud classification. GraphSAGE remains the strongest recorded model by fraud F1 and PR-AUC in the current comparison, while further validation-only threshold analysis and controlled GAT experiments may help determine whether the GAT result can be improved without compromising evaluation integrity.
