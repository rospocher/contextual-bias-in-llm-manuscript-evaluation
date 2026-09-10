# Author Metadata Affects Large Language Model Scores in Scientific Peer Review

Repository contains all the materials for the manuscript.

## Content

* analysis-outcome: contains all the results (tables and plots) produced for the various analysis conducted. In addition to the manuscript results, it also contains the effect interaction analysis (`interactions_`), and model-wise and domain-wise heterogeneity analyses (`modelwise_heterogeneity_grid_` and `domain_heterogeneity_grid_`)

* code: contains all the code for collecting the arXiv manuscript text, calling the various LLMs (under `model_generation`), and running the analysis

* configuration: the prompt used, the questionnaire used (with the different corresponding-author manipulation), the config files with the list of names and institutions used, the configuration used for building the corpus, and the arXiv papers ID used in the experiment

This work has been conducted within the [Digital Arena for Inclusive Humanities](https://daih.eu) of the University of Verona, Italy.

Full reference to the paper:
```bibtex
@article{2026dai,
	author = {Marco Rospocher},
	issn = {2731-0809},
	journal = {Discover Artificial Intelligence},
	title = {Author Metadata Affects Large Language Model Scores in Scientific Peer Review},
	year = {2026},
}