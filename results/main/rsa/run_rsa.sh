#!/bin/zsh
# Run facesim3d/modeling/rsa.py
for metric in cosine euclidean
do
    python -m facesim3d.modeling.rsa --metric ${metric} --save_corr --plot --save_plots --logger_overwrite --verbose
done
