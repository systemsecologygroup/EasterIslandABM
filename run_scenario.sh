#!/bin/bash

export sa=$1

onerun(){
        python main.py $sa $1 $2  > $sa_$1_seed$2.txt
        export folder=data/${sa}_$1_seed$2/
	echo $folder
        tar -zcvf  ${sa}_$1_seed$2.tar.gz $folder
        mv $sa_$1_seed$2.tar.gz data/packed/
        mv $sa_$1_seed$2.txt data/packed/
        rm -r $folder
}


mkdir data/packed

scenario_run(){
  export scen=$1
  for i in {1..10}; do\
          echo "Started Run with seed "$i
          onerun $scen $i &
  done
}

scenarios="homogeneous constrained full"
for s in $scenarios; do\
  	scenario_run $s
  	wait
  	echo "./run_scenario.sh finished successfully for "
	echo $s
done
