# Ensemble experiment outputs

Run `model/04_GraphSAGE_GAT_Ensemble.ipynb` from the repository root (or from `model/`) after generating the processed Elliptic graph and training the GraphSAGE/GAT baselines locally.

The notebook creates this directory and writes metrics JSON/CSV, the validation search grid, a validation F1 heatmap, a test confusion matrix, and test precision–recall and ROC curves. Generated plots and metric outputs are intentionally not tracked by Git; this README preserves the folder structure. Model checkpoints and the processed graph remain local/ignored files.
