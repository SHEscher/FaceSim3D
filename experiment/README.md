# FaceSim3D – **experiment**

    Period: November 2021 — December 2022
    Status: [finalised]

| Authors | Simon M. Hofmann           | Anthony Ciston | Abhay Koushik | Michael Gaebler      |
|---------|----------------------------|----------------|---------------|----------------------|
| Contact | simon.hofmann[@]cbs.mpg.de | ...            | ...           | gaebler[@]cbs.mpg.de |

![Last update](https://img.shields.io/badge/last_update-Nov_25,_2024-green)

***

*This folder contains mainly code to run the experiment.*

## Description of the experiment

The experiment consits of a triplet-odd-one-out task,
where participants have to identify the most dissimilar face out of three faces.
The experiment is run in two conditions:
In the 2D condition, participants see static 2D images of faces.
In the 3D condition, participants see dynamically moving faces, providing various perspectives.

## Experiment code structure

We use [`UXF 2.0`](https://github.com/immersivecognition/unity-experiment-framework), based on `Unity >= 2018.4`
for the experiment.
The corresponding code can be found in `FaceSimExp/`.
