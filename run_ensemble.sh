#!/bin/bash

export sa=$1
export sc=$2

onerun(){
        python main.py $sa $sc $1  > ${sa}_${sc}_seed$1.txt
        export folder=data/${sa}_${sc}_seed$1/
	      echo $folder
        tar -zcvf  ${sa}_${sc}_seed$1.tar.gz $folder
        mv ${sa}_${sc}_seed$1.tar.gz data/packed/
        mv ${sa}_${sc}_seed$1.txt data/packed/
        rm -rf $folder
}


for i in {11..20}; do\
        echo "Started Run with seed "$i
        onerun $i &
done

wait


