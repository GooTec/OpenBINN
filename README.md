# OpenBINN

**Biologically Informed Neural Networks: Reproducible and Interpretable Framework**

OpenBINN provides a unified folder structure and tools to train, interpret, and statistically validate Biologically Informed Neural Networks (BINNs). Our goal is to ensure that BINN-based analyses are **reproducible**, **interpretable**, and **trustworthy** through rigorous statistical methodologies.

---

## Key Features

-   **Standardized Project Layout**
    A consistent directory structure for data, models, experiments, and analyses.

-   **Model Implementation**
    Pre-built templates for building BINNs on multi-omics datasets (e.g., somatic mutations, CNV amplification/deletion).

-   **Interpretability Tools**
    Integration of explainability methods (e.g., attention- or path-informed attribution) with utility functions for visualization and summary.

-   **Statistical Validation**
    Built-in pipelines for bootstrap confidence intervals, permutation tests (gene and label), and empirical power estimation to quantify uncertainty and significance of model interpretations.

-   **Reproducibility**
    Configuration-driven experiments (YAML/JSON), version-controlled environment files, and automatic logging of parameters and random seeds.

---

## Repository Structure

```
openbinn/
├── openbinn/              # OpenBINN Module including explainer and BINN model
├── data/                  # Raw and processed omics datasets
├── biological_knowledge/  # files for implementing models with biological knowledge (e.g, Reactome, KEGG, etc.)
├── analysis/              # Interpretation scripts and statistical tests
├── scripts/               # Utilities (data preprocessing, evaluation)
├── configs/               # Default YAML/JSON config files
├── pull_data.sh           # Download datasets from remote source
├── setup.py               # Installation script
└── README.md              # Project overview and instructions
```

---

## Installation

```bash
git clone https://github.com/YourUsername/OpenBINN.git
cd OpenBINN
pip install -r requirements.txt
python setup.py develop
```

---

## Quick Start

1. **Prepare data** by running `bash pull_data.sh`. This downloads the prostate
   dataset into `../data/prostate/` relative to the repository.
2. **Generate simulation datasets** (optional) using
   `python analysis/sim_data_generation.py --beta 2 --gamma 2 --pathway_nonlinear`.
   The script loads mutation and copy number data, picks a pathway, and
   creates a nonlinear outcome according to `beta·S + gamma·S²` where `S` is the
   sum of its genes. Additional pathways scaled by δ₁ and δ₂ are included as in
   the original study. A `pca_plot.png` visualizes separation of the true
   pathway genes with the label distribution shown alongside. The intercept is
   calibrated so that the generated labels have roughly 50% prevalence, and a
   `predictor_table.csv` in each scenario lists the linear predictor, calibrated
   probability, and outcome for every sample. A logistic regression sanity check
   is run after generation and the AUC scores are stored in
   `logistic_metrics.csv`. Adjust `--beta` and `--gamma` or omit the flag for the
   linear baseline.
3. **Edit** an experiment config in `configs/`, specifying model type, dataset paths, and analysis options.
4. **Run** training and evaluation:

    ```bash

    ```

5. **Generate interpretations** and statistical reports:

    ```bash

    ```

---

## License and Attribution

This project incorporates code from the [OpenXAI](https://github.com/AI4LIFE-GROUP/OpenXAI) library, which is released under the MIT License. We retain the original copyright and license terms below:

```
MIT License

Copyright (c) 2022 AI4LIFE-GROUP

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Additionally, modifications and extensions in this repository are © 2025 \[Your Name or Organization] and are licensed under the MIT License, subject to the terms above.
