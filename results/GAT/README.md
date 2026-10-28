# GAT Results

Running `model/03_GAT_Model.ipynb` locally trains the two-layer Graph Attention Network using the shared Elliptic graph, temporal masks, class weights, and validation-fraud-F1 checkpoint-selection protocol.

The notebook saves the following artifacts here after training and final test evaluation:

- `training_loss.png`
- `validation_f1.png`
- `validation_pr_auc.png`
- `confusion_matrix.png`
- `precision_recall_curve.png`
- `roc_curve.png`
- `metrics.json`
- `metrics.csv`

The trained checkpoint is saved locally as `model/best_gat_model.pt` and the experiment metadata is saved as `model/gat_metrics.json`. Model checkpoints and the processed graph are intentionally kept out of Git.

GAT full-graph attention can require more compute and memory than GCN or GraphSAGE. The notebook uses four heads with 16 channels per head in its first layer; record any architecture changes made to fit available hardware. The Streamlit dashboard remains GCN-specific until its loader and explainer are explicitly updated and tested.