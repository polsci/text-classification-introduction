# Change log

## [1.0.4] - 2025-08-07 - migrate to Textplumber

### Changed:

- Updated notebook to use [Textplumber](https://geoffford.nz/textplumber) over the previous helper file import.

### Removed:

- Removed `text_classification_introduction_helpers.py`

## [1.0.3] - 2025-05-07 - minor fixes and documentation changes

### Changed:

- Documentation changes for maintainability
- Add acknowledgements

### Fixed

- call to lemmatizer (resolves https://github.com/polsci/text-classification-introduction/issues/1, thanks @wmk7nz)
- add auto-download of averaged_perceptron_tagger_eng for lemmatizer

## [1.0.2] - 2025-03-17

### Changed:

- Added import for typing annotations if using Python <= 3.8  
- Minor changes to instruction text  

## [1.0.1] - 2025-03-17

### Changed:

- Moved helper functions to `text_classification_introduction_helpers.py`.
- Added classes for sklearn pipeline.
- Removed Wordclouds.

### Added:

- .gitignore
- README.md
- CHANGELOG.md
- requirements.txt
- text_classification_introduction_helpers.py

## [1.0.0]

Pre-git notebook used in DIGI405 labs up to 2024.