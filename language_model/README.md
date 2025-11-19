# Pretrained ngram language models
A pretrained 1gram language model is included in this repository at `language_model/pretrained_language_models/openwebtext_1gram_lm_sil`. Pretrained 3gram and 5gram language models are available for download [here](https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq) (`languageModel.tar.gz` and `languageModel_5gram.tar.gz`) and should likewise be placed in the [`pretrained_language_models`](pretrained_language_models) directory. Note that the 3gram model requires ~60GB of RAM, and the 5gram model requires ~300GB of RAM. Furthermore, OPT 6.7b requires a GPU with at least ~12.4 GB of VRAM to load for inference.

# Dependencies
```
CMake >= 3.14
gcc >= 10.1
pytorch == 1.13.1
```

# Install language model python package
Use the `setup_lm.sh` script in the root directory of this repository to create the `b2txt25_lm` conda env and install the `lm-decoder` package to it.

# Build a new phoneme-to-words ngram language model from scratch
1. First, build binaries for building the language model:
    1. Build SRILM:
      ```bash
      cd srilm-1.7.3
      export SRILM=$PWD
      make MAKE_PIC=yes World
      make cleanest
      export PATH=$PATH:$PWD/bin/i686-m64
      ```

    2. Build openfst and other stuff:
      ```bash
      cd runtime/server/x86
      mkdir build
      cd build
      cmake ..
      make -j8
      ```

2. Build ngram LM:
  ```bash
  cd ./examples/speech/s0/
  run.sh output_dir dict_path train_corpus sil_prob formatted_train_corpus prune_threshold order
  ```


