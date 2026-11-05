#!/bin/bash

python main.py --action=train --dataset=carom --split=1 \
                --num_epochs=500 \
                --num_layers_PG=11 \
                --num_layers_R=10 \
                --num_R=0 \
                --num_f_maps=64 \
                --lr=0.0005 \
                --bce_pos_weight=5 \
                --bz=1 \
                --refinement_input=sigmoid \
                --save_best_only

