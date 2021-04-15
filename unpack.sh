#!/bin/bash

for i in {1..20}; do \
        echo ${1}${i}.tar.gz
        #mkdir ${1}/
        tar -xf ${1}${i}.tar.gz
        #rm -f ${1}${i}.tar.gz
done

