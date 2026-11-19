# Locked dev/test halves of the external test sets

`external_dev_test.csv` splits the external QM9 test set (455), PC9 (258), OE62L (836) and OE62H (574) into a
**dev** half and a locked **test** half (TASK-328, DOC-327 section 5). Built 2026-11-19 by
`scripts/lock_external_split.py` (seed 20261119): OE62H stratified by its set of non-CHONF elements, the other
sets by heavy-atom-count tercile.

| Set | dev | test |
|---|---|---|
| QM9 (external) | 228 | 227 |
| PC9 | 129 | 129 |
| OE62L | 418 | 418 |
| OE62H | 287 | 287 |

sha256 `ad70d598468e56df3ed850eb49283bf4a4bc7798699750c8ce8b3c287dd5dddb`

Rules from this date on:
- Every decision (promotion, design choice, hyperparameter, blend weight) and every few-shot label uses the
  **dev** half only.
- The **test** half is scored once per reported result, for the paper. Never tune on it.
- Results obtained before 2026-11-19 (he4, he5, the wave F few-shot study) used both halves; rerun them on
  the dev/test protocol before quoting them as held-out numbers.
- Do not regenerate this file. If it must change, record why in Stele and keep the old hash.
