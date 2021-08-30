#!/bin/bash
set -e

for i in {1..1}
do
  qsub -g gca50014 cnn-job
done
