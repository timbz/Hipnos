#!/bin/bash

cudaDataMVMult="'Cuda_Math_Plugin-mvMult.data' using 1:2:3 with lines linecolor rgb '#006400' title 'Cuda'"
cblasDataMVMult="'CBlas_Math_Plugin-mvMult.data' using 1:2:3 with lines linecolor rgb 'gold' title 'OpenBlas'"
appmlDataMVMult="'`ls AMD_APPML_Math_Plugin*-mvMult.data`' using 1:2:3 with lines linecolor rgb 'coral' title 'APPML'"

cudaDataMMMult="'Cuda_Math_Plugin-mmMult.data' using 1:2:3 with lines linecolor rgb '#006400' title 'Cuda'"
cblasDataMMMult="'CBlas_Math_Plugin-mmMult.data' using 1:2:3 with lines linecolor rgb 'gold' title 'OpenBlas'"
appmlDataMMMult="'`ls AMD_APPML_Math_Plugin*-mmMult.data`' using 1:2:3 with lines linecolor rgb 'coral' title 'APPML'"

cudaDataMMCMult="'Cuda_Math_Plugin-mmcMult.data' using 1:2:3 with lines linecolor rgb '#006400' title 'Cuda'"
cblasDataMMCMult="'CBlas_Math_Plugin-mmcMult.data' using 1:2:3 with lines linecolor rgb 'gold' title 'OpenBlas'"
appmlDataMMCMult="'`ls AMD_APPML_Math_Plugin*-mmcMult.data`' using 1:2:3 with lines linecolor rgb 'coral' title 'APPML'"

cudaDataFFT="'Cuda_Math_Plugin-fft.data' using 1:2:3 with lines linecolor rgb '#006400' title 'Cuda'"
cblasDataFFT="'CBlas_Math_Plugin-fft.data' using 1:2:3 with lines linecolor rgb 'gold' title 'OpenBlas'"
appmlDataFFT="'`ls AMD_APPML_Math_Plugin*-fft.data`' using 1:2:3 with lines linecolor rgb 'coral' title 'APPML'"

function genChart {
    echo "
            set terminal pngcairo size 700,400
            set output '$3.png'
            set xlabel \"size\"
            set ylabel \"operations\"
            set zlabel \"ms\"
            set hidden3d
                        set grid
                        set view 30,315
            splot $1, $2
    " | gnuplot
}

genChart "$cudaDataMVMult" "$cblasDataMVMult" "cudaOpenBlasMVMult"
genChart "$appmlDataMVMult" "$cblasDataMVMult" "appmlOpenBlasMVMult"
genChart "$appmlDataMVMult" "$cudaDataMVMult" "appmlCudaMVMult"

genChart "$cudaDataMMMult" "$cblasDataMMMult" "cudaOpenBlasMMMult"
genChart "$appmlDataMMMult" "$cblasDataMMMult" "appmlOpenBlasMMMult"
genChart "$appmlDataMMMult" "$cudaDataMMMult" "appmlCudaMMMult"

genChart "$cudaDataMMCMult" "$cblasDataMMCMult" "cudaOpenBlasMMCMult"
genChart "$appmlDataMMCMult" "$cblasDataMMCMult" "appmlOpenBlasMMCMult"
genChart "$appmlDataMMCMult" "$cudaDataMMCMult" "appmlCudaMMCMult"

genChart "$cudaDataFFT" "$cblasDataFFT" "cudaOpenBlasFFT"
genChart "$appmlDataFFT" "$cblasDataFFT" "appmlOpenBlasFFT"
genChart "$appmlDataFFT" "$cudaDataFFT" "appmlCudaFFT"
