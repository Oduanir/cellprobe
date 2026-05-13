"""Compatibility shim for running Geneformer inference on BioNeMo 2.7.1.

The released 10M_241113 Geneformer checkpoint serializes a Fiddle reference
to `bionemo.geneformer.api.BERTMLMLossWithReductionNoForward`, but BioNeMo
2.7.1 renamed this class to `BERTMLMLossWithReduction` (the `NoForward`
suffix was dropped). Loading the checkpoint without the alias below fails
with `AttributeError: module 'bionemo.geneformer.api' has no attribute
'BERTMLMLossWithReductionNoForward'`.

Run this in place of `infer_geneformer`; CLI args are forwarded.
"""
import bionemo.geneformer.api as api

if not hasattr(api, "BERTMLMLossWithReductionNoForward"):
    api.BERTMLMLossWithReductionNoForward = api.BERTMLMLossWithReduction

from bionemo.geneformer.scripts.infer_geneformer import geneformer_infer_entrypoint

if __name__ == "__main__":
    geneformer_infer_entrypoint()
