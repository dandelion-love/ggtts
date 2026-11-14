# OpenCC character table

`TSCharacters.txt` is OpenCC's traditional-to-simplified character table, the file the text judge folds a
chunk's text and a recogniser's transcript with (`crates/ggtts-core/src/textjudge.rs`, `Fold::t2s`). It is
copied unmodified from the PyPI package `opencc-python-reimplemented` 0.1.7 (`opencc/dictionary/`), which
carries [OpenCC](https://github.com/BYVoid/OpenCC)'s dictionary files; sha256
`6b5a0a799bea2bb22c001f635eaa3fc2904310f0c08addbff275477a80ecf09a`.

Licence: Apache-2.0. `LICENSE` and `NOTICE` are that package's, copied unmodified. They stay beside the table.
