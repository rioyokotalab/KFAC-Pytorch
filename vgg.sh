#!/bin/bash
set -e

for i in {1..16}
do
  qsub -g gca50014 vgg-job
done
