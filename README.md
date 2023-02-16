# REMIX Self-Study Instructions

## First Time Setup

- Install the public version of rust_circuit from [the GitHub repository](https://github.com/redwoodresearch/rust_circuit_public) and follow the setup instructions there.
- Install additional packages for remix with `python -m pip install -r requirements.txt`
- Download the models and data (about 700MB compressed) from [this link](https://rrserve.s3.us-west-2.amazonaws.com/remix/remix_tensors.zip) and unzip to your home directory such that you have a `~/tensors_by_hash_cache/` directory.

## Course Outline

- W0D1 - pre-course exercises on PyTorch and einops (CPU)
- D1 - Intro to Circuits (CPU)
- D2 - Build your own GPT in Circuits (CPU)
- D3 - Induction Heads Investigation (GPU preferred)
- D4a - Intro to Causal Scrubbing (GPU preferred)
- D4b - Paren Balencing (GPU preferred)
- D5 - Indirect Object Identification and Causal Scrubbing (GPU preferred)