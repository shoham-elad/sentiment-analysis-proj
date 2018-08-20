#!/bin/bash
#train/test data num_of_rows
if [$1 = 'train']
then
	python betterflow.py $2 $3
else
	if [$1 = 'test']
	then
		python testFlow.py $2 $3
	else 
		echo "not a command. please write train or test....."
	fi
fi

