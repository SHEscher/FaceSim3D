# FaceSim3D – **experiment**

    Period: November 2021 — December 2022
    Status: [finalised]

| Authors | Simon M. Hofmann           | Anthony Ciston                                                               | Abhay Koushik                                                                        | Michael Gaebler      |
|---------|:---------------------------|------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|----------------------|
| Contact | simon.hofmann[@]cbs.mpg.de | [GitHub @anfrimov](https://github.com/anfrimov "Go to personal GitHub page") | [GitHub @AbhayKoushik](https://github.com/AbhayKoushik "Go to personal GitHub page") | gaebler[@]cbs.mpg.de |

![Last update](https://img.shields.io/badge/last_update-Nov_26,_2024-green)

***

*This folder contains mainly code to run the experiment.*

## Description of the experiment

The experiment consists of a triplet-odd-one-out task,
where participants have to identify the most dissimilar face out of three faces.
The experiment is run in two conditions:
In the 2D condition, participants see static 2D images of faces.
In the 3D condition, participants see dynamically moving faces, providing various perspectives.

## Experiment setup & code

[`UXF 2.0`](https://github.com/immersivecognition/unity-experiment-framework), based on `Unity >= 2018.4`
has been used for the implementation of the face similarity judgment task (*triplet odd-one-out task*).
The corresponding code can be found in `experiment/FaceSimExp/` in the repository.
The experiment was launched as GitHub page.
Participants were recruited via [Prolific](https://www.prolific.co/),
and response data were temporarily stored on **AWS** servers.
