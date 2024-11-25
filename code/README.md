# FaceSim3D – **code**

| Period                |    Status    |           Author |
|-----------------------|:------------:|-----------------:|
| Feb, 2022 - Sep, 2024 | `in process` | Simon M. Hofmann |

![Last update](https://img.shields.io/badge/last_update-Nov_25,_2024-green)

***

## Analysis steps

### Representational Similarity Analysis (RSA)

RSA is applied on similarity matrices of face-pairs in both viewing conditions (2D, 3D) that were computed
on the following associated data:

#### Behavioral judgments

Human judgments of face similarity from the triplet odd-one-out task.

#### `VGG-Face` activation maps

Face images both in original and 3D-reconstructed form are fed into a pre-trained `VGG-Face` network.
The resulting activation maps (of all layers) were used to compute the similarity matrices.

#### Cognitive models

Modeling, i.e., predicting human similarity judgments in both conditions (2D & 3D).

##### Human-aligned `VGG-Face` activation maps

An adaptation of the VGG-Face network was alighed to human choices in the face similarity task.
Also, for this embedding model, similarity matrices were computed based on feature maps of the model.

##### Sparse embedding models: SPoSE & VICE

These models were developed for modeling human similarity judgments in similarity tasks.
See [Hebart et al. (2020)](https://www.nature.com/articles/s41562-020-00951-3),
and [Muttenthaler el al. (2022)](https://hdl.handle.net/21.11116/0000-000B-2A07-F) for
details.

#### Physical face features

Original face stimuli are from the [Chicago Face Database (CFD)](https://www.chicagofaces.org).

![CFD faces](https://www.chicagofaces.org/wp-content/uploads/2018/05/cfd_25_title4_sm.png)

Each face is associated with a large set of physical face features (e.g., length of the nose).
That is, each face can be described by a high-dimensional feature vector.
These vectors have been used to compute face similarity matrices.

#### FLAME and DECA dimensions

The 3D-reconstructed faces have corresponding FLAME and DECA dimensions.

## Code structure

The analysis code can be installed as a Python package `facesim3d`.
Check how to install it in the main [README](../README.md#research-code-facesim3d-as-python-package).

### Testing

```shell
pytest \              # (standard run through the tests)
  --cov \             # (check how much (%) of the code is tested already)
  --cov-report=html   # (to explore which parts of the code are not tested yet)
```

## COPYRIGHT/LICENSE

See [LICENSE](../LICENSE) for details on usage and citation.
