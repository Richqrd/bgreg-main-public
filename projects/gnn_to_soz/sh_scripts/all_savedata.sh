#!/bin/bash
array=("jh101" "jh102" "jh103" "jh108"
       "pt01" "pt10" "pt11" "pt12" "pt13" "pt14" "pt16" "pt2" "pt3" "pt6" "pt7" "pt8"
       "umf001" "ummc001" "ummc002" "ummc003" "ummc004" "ummc005" "ummc006" "ummc007" "ummc009")
for i in "${array[@]}"
do
	bash savedata.sh "$i"
	sleep 5
done