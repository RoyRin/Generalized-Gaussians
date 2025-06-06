#!/bin/bash

#set -x 
poetry shell
# specificy GPU id
for i in {0..3};
do
	echo $i
	echo "----"
	a=`expr $i % 4`
	echo $a
	j=`expr $i + 4 `
	echo $j
	echo "=="
	CUDA_VISIBLE_DEVICES=$a python beta_image_classifier.py $i && CUDA_VISIBLE_DEVICES=$a python beta_image_classifier.py $j &
done

