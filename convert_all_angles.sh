#!/bin/bash

num_ang=$1
d_ang=$2

for ((d=0; d<$num_ang; d++))
  do
    python pyDVS/sequence_to_spikes.py $(($d_ang*$d))
  done
