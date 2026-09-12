## Library

反応ファイルの探索と DMF/UMA 計算は `batch_colab_reaction` から利用できます。

```python
from batch_colab_reaction import scan_reaction_files, run_all_reactions

pairs = scan_reaction_files(".")
results = run_all_reactions(pairs, predictor, output_dir="results")
```

入力ファイル名は `name_reactant.xyz` / `name_product.xyz`（または `.pdb`）です。
`scan_reaction_files` はペア不足・重複・形式不一致を `ValueError` として通知します。
