# The Same Paper, Different Verdict: Contextual Bias in LLM-Based Manuscript Evaluation

Repository contains all the materials for the manuscript

## Content

* analysis-outcome: contains all the results (tables and plots) produced for the various analysis conducted. In addition to the manuscript results, it also contains the effect interaction analysis (`interactions_`), leave one token one analysis (`loto_`), and model-wise and domain-wise heterogeneity analyses (`modelwise_heterogeneity_grid_` and `domain_heterogeneity_grid_`), for both experimental settings

* code: contains all the code for collecting the arXiv manuscript text, calling the various LLMs (under `model_generation`), running the analysis for each experimental setting, and running the comparison between them.

* configuration: the prompt used, the questionnaire used (with the different corresponding-author manipulation), the config files with the list of names and institutions used, the configuration used for building the corpus, and the arXiv papers ID used in the experiment