#!/usr/bin/env bash

script_dir=`dirname $(realpath $0)`

offload=1

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

    #if [[ ! "$d" == "srad_v2" ]]
    #then
    #    continue
    #fi

    # print project name
    #printf "%-20s" $d

    if [[ ! -f "verify" ]]
    then
        echo "Not support"
        continue
    fi

    make clean &> /dev/null


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


    # measure time and check timeout
    timeout 20s ./run &> /dev/null
    ret=$?

    if [[ ! $ret -eq 0 ]]
    then
        echo "Fail"
        make clean &> /dev/null
        continue
    fi

    ./verify &> /dev/null
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
