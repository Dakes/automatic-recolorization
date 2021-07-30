#!/bin/sh

srcDir=$1
destDir=$2
opts="-vf crop=1064:800:492:0,scale=640:480"

for filename in "$srcDir"/*leftImg8bit.png; do
    basePath=${filename%.*}
    baseName=${basePath##*/}
    ffmpeg -i "$filename" $opts "$destDir"/"$baseName"."png"

done



