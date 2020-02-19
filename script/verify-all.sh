#!/usr/bin/env bash

script_dir=`dirname $(realpath $0)`

# Reset
Color_Off='\033[0m'       # Text Reset

# Regular Colors
Black='\033[0;30m'        # Black
Red='\033[0;31m'          # Red
Green='\033[0;32m'        # Green
Yellow='\033[0;33m'       # Yellow
Blue='\033[0;34m'         # Blue
Purple='\033[0;35m'       # Purple
Cyan='\033[0;36m'         # Cyan
White='\033[0;37m'        # White

offload=1
#offload=0

bulk=1

if [[ $offload -eq 1 ]]
then
    printf "${Red}OFFLOAD ${Color_Off}Enabled\n"
fi

if [[ $bulk -eq 1 ]]
then
    printf "${Yellow}BULK ${Color_Off}Enabled\n"
fi

printf "$Color_Off"

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
        echo "No verify"
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
            echo "No offload"
            continue
        fi
        make OFFLOAD=1 &> /dev/null
    fi
    ret=$?

    if [[ ! $ret -eq 0 ]]
    then
        echo -n "[CE] "
    fi


    # measure time and check timeout
    #timeout 360s ./run &> /dev/null
    #ret=$?

    #if [[ ! $ret -eq 0 ]]
    #then
    #    echo "Fail"
    #    make clean &> /dev/null
    #    continue
    #fi

    if [[ $bulk -eq 1 ]]
    then
        OMP_BULK=1 timeout 3000s ./verify &> /dev/null
        ret=$?
        if [[ ! $ret -eq 0 ]]
        then
            echo -n "[TL]"
        fi
    else
        timeout 360s ./verify &> /dev/null
        ret=$?
    fi


    if [[ $ret -eq 0 ]]
    then
        echo "Pass"
    else
        echo "Fail"
    fi

    #exec 1>&3 2>&4

    make clean &> /dev/null
done
