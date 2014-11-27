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

if $runMVMult
then
    echo "
            set terminal pngcairo size 700,400
            set output 'mvMult.png'
            set grid
            set key right bottom
            set logscale y
            set xlabel 'Matrixgröße' font 'Times,13'
            set ylabel 'Berechnungszeit [ms]' font 'Times,13'
            plot    'CBlas_Math_Plugin-netlib-mvMult.data' using 1:2 with lines linecolor rgb 'red' title 'netlib', \
                    'CBlas_Math_Plugin-ATLAS-mvMult.data' using 1:2 with lines linecolor rgb 'blue' title 'ATLAS', \
                    'CBlas_Math_Plugin-OpenBLAS-mvMult.data' using 1:2 with lines linecolor rgb 'gold' title 'OpenBLAS', \
                    'GSL_Math_Plugin-mvMult.data' using 1:2 with lines linecolor rgb '#FF00FF' title 'GSL', \
                    'Cuda_Math_Plugin-mvMult.data' using 1:2 with lines linecolor rgb '#006400' title 'Cuda', \
                    'Cuda_Math_Plugin_(Single_Precision)-mvMult.data' using 1:2 with lines linecolor rgb 'greenyellow' title 'Cuda SP', \
                    'AMD_ACML_Math_Plugin-mvMult.data' using 1:2 with lines linecolor rgb 'black' title 'ACML', \
                    '`ls AMD_APPML_Math_Plugin*-mvMult.data`' using 1:2 with lines linecolor rgb 'coral' title 'APPML'
    " | gnuplot
fi

if $runMMMult
then
    echo "
            set terminal pngcairo size 700,400
            set output 'mmMult.png'
            set grid
            set key right bottom
            set logscale y
            set xlabel 'Matrixgröße' font 'Times,13'
            set ylabel 'Berechnungszeit [ms]' font 'Times,13'
            plot    'CBlas_Math_Plugin-netlib-mmMult.data' using 1:2 with lines linecolor rgb 'red' title 'netlib', \
                    'CBlas_Math_Plugin-ATLAS-mmMult.data' using 1:2 with lines linecolor rgb 'blue' title 'ATLAS', \
                    'CBlas_Math_Plugin-OpenBLAS-mmMult.data' using 1:2 with lines linecolor rgb 'gold' title 'OpenBLAS', \
                    'GSL_Math_Plugin-mmMult.data' using 1:2 with lines linecolor rgb '#FF00FF' title 'GSL', \
                    'Cuda_Math_Plugin-mmMult.data' using 1:2 with lines linecolor rgb '#006400' title 'Cuda', \
                    'Cuda_Math_Plugin_(Single_Precision)-mmMult.data' using 1:2 with lines linecolor rgb 'greenyellow' title 'Cuda SP', \
                    'AMD_ACML_Math_Plugin-mmMult.data' using 1:2 with lines linecolor rgb 'black' title 'ACML', \
                    '`ls AMD_APPML_Math_Plugin*-mmMult.data`' using 1:2 with lines linecolor rgb 'coral' title 'APPML'
    " | gnuplot
fi

if $runAdd
then
    echo "
            set terminal pngcairo size 700,400
            set output 'add.png'
            set grid
            set key right bottom
            set logscale y
            set xlabel 'Matrixgröße' font 'Times,13'
            set ylabel 'Berechnungszeit [ms]' font 'Times,13'
            plot    'CBlas_Math_Plugin-netlib-add.data' using 1:2 with lines linecolor rgb 'red' title 'netlib', \
                    'CBlas_Math_Plugin-ATLAS-add.data' using 1:2 with lines linecolor rgb 'blue' title 'ATLAS', \
                    'CBlas_Math_Plugin-OpenBLAS-add.data' using 1:2 with lines linecolor rgb 'gold' title 'OpenBLAS', \
                    'GSL_Math_Plugin-add.data' using 1:2 with lines linecolor rgb '#FF00FF' title 'GSL', \
                    'Cuda_Math_Plugin-add.data' using 1:2 with lines linecolor rgb '#006400' title 'Cuda', \
                    'Cuda_Math_Plugin_(Single_Precision)-add.data' using 1:2 with lines linecolor rgb 'greenyellow' title 'Cuda SP', \
                    'AMD_ACML_Math_Plugin-add.data' using 1:2 with lines linecolor rgb 'black' title 'ACML', \
                    '`ls AMD_APPML_Math_Plugin*-add.data`' using 1:2 with lines linecolor rgb 'coral' title 'APPML'
    " | gnuplot
fi

if $runCMMMult
then
    echo "
            set terminal pngcairo size 700,400
            set output 'mmcMult.png'
            set grid
            set key right bottom
            set logscale y
            set xlabel 'Matrixgröße' font 'Times,13'
            set ylabel 'Berechnungszeit [ms]' font 'Times,13'
            plot    'CBlas_Math_Plugin-mmcMult.data' using 1:2 with lines linecolor rgb 'blue' title 'ATLAS', \
                    'GSL_Math_Plugin-mmcMult.data' using 1:2 with lines linecolor rgb '#FF00FF' title 'GSL', \
                    'Cuda_Math_Plugin-mmcMult.data' using 1:2 with lines linecolor rgb '#006400' title 'Cuda', \
                    'Cuda_Math_Plugin_(Single_Precision)-mmcMult.data' using 1:2 with lines linecolor rgb 'greenyellow' title 'Cuda SP', \
                    'AMD_ACML_Math_Plugin-mmcMult.data' using 1:2 with lines linecolor rgb 'black' title 'ACML', \
                    '`ls AMD_APPML_Math_Plugin*-mmcMult.data`' using 1:2 with lines linecolor rgb 'coral' title 'APPML'
    " | gnuplot
fi

if $runFFT
then
    echo "
            set terminal pngcairo size 700,400
            set output 'fft.png'
            set grid
            set key right bottom
            set logscale y
            set xlabel 'Matrixgröße' font 'Times,13'
            set ylabel 'Berechnungszeit [ms]' font 'Times,13'
            plot    'CBlas_Math_Plugin-fft.data' using 1:2 with lines linecolor rgb 'blue' title 'FFTW', \
                    'GSL_Math_Plugin-fft.data' using 1:2 with lines linecolor rgb '#FF00FF' title 'GSL', \
                    'Cuda_Math_Plugin-fft.data' using 1:2 with lines linecolor rgb '#006400' title 'Cuda', \
                    'Cuda_Math_Plugin_(Single_Precision)-fft.data' using 1:2 with lines linecolor rgb 'greenyellow' title 'Cuda SP', \
                    'AMD_ACML_Math_Plugin-fft.data' using 1:2 with lines linecolor rgb 'black' title 'ACML', \
                    '`ls AMD_APPML_Math_Plugin*-fft.data`' using 1:2 with lines linecolor rgb 'coral' title 'APPML'
    " | gnuplot
fi

if $runCpy
then
    echo "
            set terminal pngcairo size 700,400
            set output 'cpy.png'
            set grid
            set key right bottom
            set xlabel 'Matrixgröße' font 'Times,13'
            set ylabel 'Berechnungszeit [ms]' font 'Times,13'
            plot    'Cuda_Math_Plugin-cpy.data' using 1:2 with lines linecolor rgb '#006400' title 'Cuda', \
                    'Cuda_Math_Plugin_(Single_Precision)-cpy.data' using 1:2 with lines linecolor rgb 'greenyellow' title 'Cuda SP', \
                    '`ls AMD_APPML_Math_Plugin*-cpy.data`' using 1:2 with lines linecolor rgb 'coral' title 'APPML'
    " | gnuplot
fi
