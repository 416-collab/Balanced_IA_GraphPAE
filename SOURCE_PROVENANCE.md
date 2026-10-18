# Source provenance

Each top-level source `.py` in this staging package is copied byte-for-byte from `revision_results/ALL_REVISION_20261018_031751_881688/` in the uploaded reviewer archive, rather than the outdated root files.

Configs:
- `config/actor.yaml` <= `revision_results/ALL_REVISION_20261018_031751_881688/config/actor.yaml`
- `config/chameleon.yaml` <= `revision_results/ALL_REVISION_20261018_031751_881688/config/chameleon.yaml`
- `config/citeseer.yaml` <= `revision_results/ALL_REVISION_20261018_031751_881688/config/citeseer.yaml`
- `config/cora.yaml` <= `revision_results/ALL_REVISION_20261018_031751_881688/config/cora.yaml`
- `config/squirrel.yaml` <= `revision_results/ALL_REVISION_20261018_031751_881688/config/squirrel.yaml`
- `config/arxiv-year-official.yaml` <= `config/arxiv-year-official.yaml`
- `config/penn94.yaml` <= `revision_results/PENN94_BF16_INFERENCE_20261014_193038_498135/config/penn94.yaml`


The archive included earlier root configs with incompatible layer count or hidden dimensions for Actor, CiteSeer, Squirrel, and Penn94. Those are NOT used here. The verified 5-seed result files are preserved separately in the `evidence/` tree.
