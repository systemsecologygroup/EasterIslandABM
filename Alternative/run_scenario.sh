#!/bin/bash

echo "ALTERNATIVE SCENARIO"

export sa=$1
mkdir data
mkdir data/packed

scenario_run(){
  ./run_ensemble.sh $1 $2
}

scenarios="homogeneous constrained full"
for sc in $scenarios; do\
  echo "Starting runs for the following settings with different seeds"
  echo "Sensitivity parameters: "$sa
  echo "scenario params: "$sc
  scenario_run $sa $sc
  wait
  echo "./run_scenario.sh finished successfully for "
	echo $sc
done
