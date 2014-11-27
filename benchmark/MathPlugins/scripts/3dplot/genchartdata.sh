#!/bin/bash

date

runMVMult=false
runMMMult=false
runCMMMult=false
runFFT=false

if [ -z "$1" ]
then
    runMVMult=true
    runMMMult=true
    runCMMMult=true
    runFFT=true
    runCpy=true
else
    if [[ "$1" == *mvMult* ]]
    then
        runMVMult=true
    fi
    if [[ "$1" == *mmMult* ]]
    then
        runMMMult=true
    fi
    if [[ "$1" == *mmcMult* ]]
    then
        runCMMMult=true
    fi
    if [[ "$1" == *fft* ]]
    then
        runFFT=true
    fi
fi

declare -A currentMatrixSizePerPlugin

function parseTestData {
    cat testdata.xml | grep BenchmarkResult | while read line ; do

        metric=`echo $line | sed 's/.*metric="\([^"]*\)".*/\1/'`
        tag=`echo $line | sed 's/.*tag="\([^"]*\)".*/\1/'`
        total=`echo $line | sed 's/.*value="\([^"]*\)".*/\1/'`
        iterations=`echo $line | sed 's/.*iterations="\([^"]*\)".*/\1/'`

        value=`echo "scale=4; $total/$iterations" | bc`
        plugin=`echo $tag | sed 's/.*plugin=\[\([^\[]*\)\].*/\1/'`
        matrixSize=`echo $tag | sed 's/.*matrixSize=\[\([^\[]*\)\].*/\1/'`
        operations=`echo $tag | sed 's/.*operations=\[\([^\[]*\)\].*/\1/'`

        if [ -z "$1" ]
        then
            filename=`echo $plugin.data | sed 's/\s/_/g'`
        else
            filename=`echo $plugin-$1.data | sed 's/\s/_/g'`
        fi

        if [ -z "${currentMatrixSizePerPlugin[$plugin]}" ]
        then
                echo -e "#matrixSize\toperations\tvalue" > $filename
                currentMatrixSizePerPlugin[$plugin]="$matrixSize"
        fi

        if [ "$matrixSize" -ne "${currentMatrixSizePerPlugin[$plugin]}" ]
        then
            echo >> $filename
                    currentMatrixSizePerPlugin[$plugin]="$matrixSize"
        fi

        echo -e "$matrixSize\t$operations\t$value" >> $filename
    done
}

echo "Using OpenBlas"
sudo update-alternatives --set libblas.so.3gf `update-alternatives --list libblas.so.3gf | grep openblas`

if $runMVMult
then
    echo
    echo "*** Benchmarking matrix vector mult"
    echo
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixVecMult3d > testdata.xml
    echo "Parsing data ..."
    parseTestData "mvMult"
fi

if $runMMMult
then
    echo
    echo "*** Benchmarking matrix matrix mult"
    echo
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixMatrixMult3d > testdata.xml
    echo "Parsing data ..."
    parseTestData "mmMult"
fi

if $runCMMMult
then
    echo
    echo "*** Benchmarking component wise matrix mult"
    echo
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixMatrixComponentWiseMult3d > testdata.xml
    echo "Parsing data ..."
    parseTestData "mmcMult"
fi

if $runFFT
then
    echo
    echo "*** Benchmarking 2D fft"
    echo
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixFFT3d > testdata.xml
    echo "Parsing data ..."
    parseTestData "fft"
fi

date

#set blas lib back to to netlib (if openblas is loaded acml has problems with multithreading)
sudo update-alternatives --set libblas.so.3gf /usr/lib/libblas/libblas.so.3gf
