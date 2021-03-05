#!/bin/bash

export sa=$1

onerun(){
        python main.py $sa $1 $2  > ${sa}_$1_seed$2.txt
        export folder=data/${sa}_${1}_seed${2}/
        tar -zcvf  ${sa}_${1}_seed${2}.tar.gz ${folder}
        mv ${sa}_${1}_seed${2}.tar.gz data/packed/
        mv ${sa}_${1}_seed${2}.txt data/packed/
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

for scen in {homogeneous, constrained, full}; do\
  scenario_run() $scen
  wait
  echo "A ./run_scenario.sh finished successfully"
done