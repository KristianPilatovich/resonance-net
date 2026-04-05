#!/bin/bash
cd "/home/kristian/Рабочий стол/AI/resonance_net/build"
exec ./resonance_net train \
  --data "../data/train.bin" \
  --steps 100000 \
  --batch 32 \
  --lr 0.001 \
  --log 5000 \
  --save "../checkpoints" \
  >> "../train_100k.log" 2>&1
