#!/bin/bash

runMVMult=false
runMMMult=false
runCMMMult=false
runFFT=false
runCpy=false
runAdd=false

if [ -z "$1" ]
then
    runMVMult=true
    runMMMult=true
    runCMMMult=true
    runFFT=true
    runCpy=true
    runAdd=true
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
    if [[ "$1" == *cpy* ]]
    then
        runCpy=true
    fi
    if [[ "$1" == *add* ]]
    then
        runAdd=true
    fi
fi

export OMP_NUM_THREADS=2
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
                echo -e "#matrixSize\tvalue" > $filename
                currentMatrixSizePerPlugin[$plugin]="$matrixSize"
        fi
        echo -e "$matrixSize\t$value" >> $filename

    done
}

if $runMVMult
then
    echo
    echo "*** Benchmarking matrix vector mult"
    echo
    update-alternatives --list libblas.so.3gf | while read blasPath ; do

            blasLib="netlib"
            if [[ "$blasPath" == *atlas* ]]
            then
              blasLib="ATLAS";
            fi
            if [[ "$blasPath" == *openblas* ]]
            then
              blasLib="OpenBLAS";
            fi

            echo "Using $blasLib"
            sudo update-alternatives --set libblas.so.3gf $blasPath

            echo "Running benchmark ..."
            ../../mathpluginsbenchmark -xml matrixVecMultCBlas > testdata.xml

            echo "Parsing data ..."
            parseTestData $blasLib"-mvMult"

    done

    #set blas lib back to to netlib (if openblas is loaded acml has problems with multithreading)
    sudo update-alternatives --set libblas.so.3gf /usr/lib/libblas/libblas.so.3gf

    echo "Using GSL"
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixVecMultGSL > testdata.xml
    echo "Parsing data ..."
    parseTestData "mvMult"

    echo "Using ACML"
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixVecMultACML > testdata.xml
    echo "Parsing data ..."
    parseTestData "mvMult"

    echo "Using Cuda"
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixVecMultCuda > testdata.xml
    echo "Parsing data ..."
    parseTestData "mvMult"

    echo "Using APPML"
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixVecMultAPPML > testdata.xml
    echo "Parsing data ..."
    parseTestData "mvMult"
fi

if $runMMMult
then
    echo
    echo "*** Benchmarking matrix matrix mult"
    echo
    update-alternatives --list libblas.so.3gf | while read blasPath ; do

            blasLib="netlib"
            if [[ "$blasPath" == *atlas* ]]
            then
              blasLib="ATLAS";
            fi
            if [[ "$blasPath" == *openblas* ]]
            then
              blasLib="OpenBLAS";
            fi

            echo "Using $blasLib"
            sudo update-alternatives --set libblas.so.3gf $blasPath

            echo "Running benchmark ..."
            ../../mathpluginsbenchmark -xml matrixMatrixMultCBlas > testdata.xml

            echo "Parsing data ..."
            parseTestData $blasLib"-mmMult"

    done

    #set blas lib back to to netlib (if openblas is loaded acml has problems with multithreading)
    sudo update-alternatives --set libblas.so.3gf /usr/lib/libblas/libblas.so.3gf

    echo "Using GSL"
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixMatrixMultGSL > testdata.xml
    echo "Parsing data ..."
    parseTestData "mmMult"

    echo "Using ACML"
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixMatrixMultACML > testdata.xml
    echo "Parsing data ..."
    parseTestData "mmMult"

    echo "Using Cuda"
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixMatrixMultCuda > testdata.xml
    echo "Parsing data ..."
    parseTestData "mmMult"

    echo "Using APPML"
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixMatrixMultAPPML > testdata.xml
    echo "Parsing data ..."
    parseTestData "mmMult"
fi

if $runCMMMult
then
    echo
    echo "*** Benchmarking component wise matrix matrix mult"
    echo

    #set blas lib back to to netlib (if openblas is loaded acml has problems with multithreading)
    sudo update-alternatives --set libblas.so.3gf /usr/lib/libblas/libblas.so.3gf

    echo "Using CBlas"
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixMatrixComponentWiseMultCBlas > testdata.xml

    echo "Parsing data ..."
    parseTestData "mmcMult"

    echo "Using GSL"
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixMatrixComponentWiseMultGSL > testdata.xml
    echo "Parsing data ..."
    parseTestData "mmcMult"

    echo "Using ACML"
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixMatrixComponentWiseMultACML > testdata.xml
    echo "Parsing data ..."
    parseTestData "mmcMult"

    echo "Using Cuda"
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixMatrixComponentWiseMultCuda > testdata.xml
    echo "Parsing data ..."
    parseTestData "mmcMult"

    echo "Using APPML"
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixMatrixComponentWiseMultAPPML > testdata.xml
    echo "Parsing data ..."
    parseTestData "mmcMult"
fi

if $runAdd
then
    echo
    echo "*** Benchmarking matrix matrix add"
    echo
    update-alternatives --list libblas.so.3gf | while read blasPath ; do

            blasLib="netlib"
            if [[ "$blasPath" == *atlas* ]]
            then
              blasLib="ATLAS";
            fi
            if [[ "$blasPath" == *openblas* ]]
            then
              blasLib="OpenBLAS";
            fi

            echo "Using $blasLib"
            sudo update-alternatives --set libblas.so.3gf $blasPath

            echo "Running benchmark ..."
            ../../mathpluginsbenchmark -xml matrixAddCBlas > testdata.xml

            echo "Parsing data ..."
            parseTestData $blasLib"-add"

    done

    #set blas lib back to to netlib (if openblas is loaded acml has problems with multithreading)
    sudo update-alternatives --set libblas.so.3gf /usr/lib/libblas/libblas.so.3gf

    echo "Using GSL"
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixAddGSL > testdata.xml
    echo "Parsing data ..."
    parseTestData "add"

    echo "Using ACML"
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixAddACML > testdata.xml
    echo "Parsing data ..."
    parseTestData "add"

    echo "Using Cuda"
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixAddCuda > testdata.xml
    echo "Parsing data ..."
    parseTestData "add"

    echo "Using APPML"
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixAddAPPML > testdata.xml
    echo "Parsing data ..."
    parseTestData "add"
fi

if $runFFT
then
    echo
    echo "*** Benchmarking 2D FFT"
    echo

    #set blas lib back to to netlib (if openblas is loaded acml has problems with multithreading)
    sudo update-alternatives --set libblas.so.3gf /usr/lib/libblas/libblas.so.3gf

    echo "Using FFTW"
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixFFTCBlas > testdata.xml
    echo "Parsing data ..."
    parseTestData "fft"

    echo "Using GSL"
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixFFTGSL > testdata.xml
    echo "Parsing data ..."
    parseTestData "fft"

    echo "Using ACML"
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixFFTACML > testdata.xml
    echo "Parsing data ..."
    parseTestData "fft"

    echo "Using Cuda"
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixFFTCuda > testdata.xml
    echo "Parsing data ..."
    parseTestData "fft"

    echo "Using APPML"
    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml matrixFFTAPPML > testdata.xml
    echo "Parsing data ..."
    parseTestData "fft"
fi

if $runCpy
then
    echo
    echo "*** Benchmarking Memcpy from host to device"
    echo

    echo "Running benchmark ..."
    ../../mathpluginsbenchmark -xml copyHostToDevice > testdata.xml
    echo "Parsing data ..."
    parseTestData "cpy"
fi
