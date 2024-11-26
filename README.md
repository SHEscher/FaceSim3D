# FaceSim3D 🗿

*Testing the effect of dynamic 3D viewing conditions on face similarity perception*

![Last update](https://img.shields.io/badge/last_update-Nov_26,_2024-green)
![version](https://img.shields.io/badge/version-v.1.0.0-blue)
[![demo](https://img.shields.io/badge/pretty-docs-violet)](https://shescher.github.io/FaceSim3D/ "Go to the project's documentation page")

## Project description

To test the effect of space and time on face similarity judgments,
we conducted an online experiment using a triplet odd-one-out task
in a static 2D and a dynamic 3D condition.
We then trained sparse and deep computational encoding models of human face similarity judgments
to investigate the latent representations that underlie their predictions.

## Analysis of face similarity judgments in 2D and 3D

`code/` contains the code for data quality control, preprocessing, modeling, and analysis.

### Research code `facesim3d` as Python package

The code for data quality control, preprocessing, modeling and further analysis
is available as the Python package `facesim3d`.
The package has been implemented in `Python 3.10.4` (no other versions have been tested).

Ideally, create a virtual environment (e.g., `conda`) to install the package:

```shell
conda create -n face_3.10 python=3.10.4
```

Activate the environment:

```shell
conda activate face_3.10
```

#### Install the package

```shell
# Go to root folder of FaceSim3D
cd FaceSim3D/
pip install -e .
```

Two computational models (SPosE, VICE) that were used require additional packages
that can be installed with the following command:

```shell
pip install -e ".[spose,vice]"
```

## Experiment: Face similarity judgments in 2D and 3D
Participants had the task to identify the dissimilar face out of three faces in a triplet odd-one-out task.

`experiment/` contains the code for the experimental setup.
The experiment is based on [`UXF 2.0`](https://github.com/immersivecognition/unity-experiment-framework),
and `Unity`.

## Results

Results are summarized in form of a Jupyter notebook in `code/notebooks/`.
Additionally, some of the analysis outputs are stored in the folder `results/`.

## Research data

*Link to the research data will be offered soon.*

To reproduce the analysis, the data should be stored in the `data/` folder.

## Citation

If you use this code or data, please cite the following paper
([Hofmann et al. Human-aligned deep and sparse encoding models of dynamic 3D face similarity perception. *PsyArXiv*. 2024](https://doi.org/10.31234/osf.io/f62pw)):

```bibtex
@article{hofmannHumanalignedDeepSparse2024,
    title={Human-aligned deep and sparse encoding models of dynamic {3D} face similarity perception},
    author={Hofmann, Simon M. and Ciston, Anthony and Koushik Abhay and Klotzsche, Felix and Hebart, Martin N. and Müller, Klaus-Robert and Villringer, Arno and Scherf, Nico and Hilsmann, Anna and Nikulin, Vadim V. and Gaebler, Michael},
    journal={PsyArXiv},
    doi={10.31234/osf.io/f62pw},
    year={2024},
}
```

### References to former literature

`literature/` comprises `*.bib` file(s) that contain relevant literature, e.g., as listed in the FaceSim3D publication.

## Contributors/Collaborators

[Simon M. Hofmann*](https://bsky.app/profile/smnhfmnn.bsky.social "Follow on Bluesky"),
[Anthony Ciston](https://github.com/anfrimov "On GitHub"),
[Abhay Koushik](https://www.abhaykoushik.com "Personal webpage"),
[Felix Klotzsche](https://bsky.app/profile/flxklotz.bsky.social "Follow on Bluesky"),
[Martin N. Hebart](http://martin-hebart.de "Personal webpage"),
[Klaus-Robert Müller](https://web.ml.tu-berlin.de/author/prof.-dr.-klaus-robert-muller/ "Institute's webpage"),
[Arno Villringer](https://www.cbs.mpg.de/employees/villringer "Institute's webpage"),
[Nico Scherf](https://scholar.google.de/citations?user=mRKOyBIAAAAJ&hl=de "On Google Scholar"),
[Anna Hilsmann](https://iphome.hhi.de/hilsmann/index.htm "Institute's webpage"),
[Vadim V. Nikulin](https://www.cbs.mpg.de/employees/nikulin "Institute's webpage"),
[Michael Gaebler](https://www.michaelgaebler.com "Personal webpage")

This study was conducted at the [Max Planck Institute for Human Cognitive and Brain Sciences](https://www.cbs.mpg.de/en "Go the institute website")
as part of the [NEUROHUM project](https://neurohum.cbs.mpg.de "Go the project site").

[![NEUROHUM Logo](https://neurohum.cbs.mpg.de/assets/institutes/headers/cbsneurohum-desktop-en-cc55f3158c5428ca969719e99df1c4f636a0662c1c42e409d476328092106060.svg)](https://neurohum.cbs.mpg.de "Go the project site")

*\* corresponding author*
