#!/usr/bin/env bash

origin=`pwd`
script_dir=`dirname $(realpath $0)`

offload=0

if [[ $offload -eq 1 ]]
then
    echo "OFFLOAD mode"
fi

cd $script_dir

cd ../openmp

omp_dir=`pwd`

for d in `ls | sort -V`
do
    cd $omp_dir

    if [[ -f $d ]]
    then
        continue
    fi

    cd $d

    make clean &> /dev/null

    # print project name
    #printf "%-20s" $d

    if [[ $offload -eq 0 ]]
    then
        make  &> /dev/null
    else
        # Check OMP
        grep -r OMP_OFFLOAD &> /dev/null
        ret=$?

        if [[ $ret -ne 0 ]]
        then
            echo "Not support"
            continue
        fi
        make OFFLOAD=1 &> /dev/null
    fi

    timeout 10s ./verify &> /dev/null
    ret=$?

    if [[ $ret -eq 0 ]]
    then
        echo "Pass"
    else
        echo "Fail"
    fi

    #exec 1>&3 2>&4

    make clean &> /dev/null
done




cd $origin
