# Third-party material and licensing status

This file records attribution and known provenance. It does not grant new rights
or change the license of any included material. A project-wide software license
has not yet been declared for the original Python implementation.

## Crane paper and derived extracts

- **Work:** *Crane: Inter-Layer Scheduling Framework for DNN Inference and Training
  Co-Support on Tiled Architecture*.
- **Authors:** Yu Gong, Lingyi Huang, Haodong Chang, Rongjian Liang, Cheng Yang,
  Zhexiang Tang, Jiang Hu and Bo Yuan.
- **Publication:** MICRO 2025, pp. 1250–1263;
  [DOI: 10.1145/3725843.3756023](https://doi.org/10.1145/3725843.3756023).
- **Locations:** `include/crane_paper.pdf`, `include/crane_paper_extracted.txt`
  and paper extracts under `outputs/`.
- **Notice:** the included PDF states that the work is licensed under
  [Creative Commons Attribution–NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

The paper is included as reference material. Its license does not automatically
apply to this repository's code. The extraction and analysis files are local
working references, not an additional publication by the paper's authors.

## SET network reference files

- **Upstream:** [SET-Scheduling-Project/SET-ISCA2023](https://github.com/SET-Scheduling-Project/SET-ISCA2023).
- **Recorded source revision:** `a7bd73912f58d9fea10fadab693eebd4e6e3054d`.
- **Recorded sync date:** 2026-03-06, from [`src/nns/SOURCE.txt`](src/nns/SOURCE.txt).
- **Locations:** `src/nns/*.cpp`, `include/network.h`, `include/nns/nns.h`.
- **Purpose:** network graph/shape reference definitions, not trained weights
  and not the Crane authors' released implementation.

The checked-in source record identifies the copied C++ network files. The header
provenance is also described in the archived
[source explanation](outputs/docs/references/official_nns_data_source_explanation.txt).
No license accompanies this copied subset, and GitHub's upstream license endpoint
did not identify a license at the time of this documentation review. That absence
must not be interpreted as a permissive license; upstream terms/permission need
to be established before claiming redistribution rights for these files.

## Runtime dependencies

NumPy, OR-Tools/SCIP, pypdf and their dependencies are installed through pip and
retain their own licenses. `requirements.txt` and `constraints.txt` record the
dependency set. Locally generated virtual environments and bytecode are excluded
from new commits; historical Git revisions may still contain such files.
