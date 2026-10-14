# Training corpora

The abuse detector is fine-tuned on a merge of three public datasets. **They are not included in this repository.** Download each from its original publisher and follow its license/terms of use.

| Dataset | Reference |
|---|---|
| ToxiCN | Lu, J., Xu, B., Zhang, X., et al. Facilitating Fine-grained Detection of Chinese Toxic Language: Hierarchical Taxonomy, Resources, and Benchmarks. *Proceedings of the 61st Annual Meeting of the ACL*, Toronto, Canada, 2023, vol. 1, pp. 16235–16250. |
| COLD | Deng, J., Zhou, J., Sun, H., et al. COLD: A Benchmark for Chinese Offensive Language Detection. *Proceedings of EMNLP 2022*, Abu Dhabi, UAE, pp. 11580–11599. |
| SWSR | Jiang, A., Yang, X., Liu, Y., et al. SWSR: A Chinese Dataset and Lexicon for Online Sexism Detection. *Online Social Networks and Media*, 2022, 27:100182. |

## Expected format for `train.py`

A CSV with two columns:

| text | label |
|---|---|
| (a sentence) | 0 = not abusive, 1 = abusive |

Map each source dataset's own labels to this binary scheme before merging, and keep a note of how you did it (which sub-labels count as abusive), since it affects results.
