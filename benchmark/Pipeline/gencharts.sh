#set blas lib back to to ATLAS
sudo update-alternatives --set libblas.so.3gf /usr/lib/atlas-base/atlas/libblas.so.3gf

declare -A currentOperationsPerPlugin

function parseTestData {
    cat testdata.xml | grep BenchmarkResult | while read line ; do

        metric=`echo $line | sed 's/.*metric="\([^"]*\)".*/\1/'`
        tag=`echo $line | sed 's/.*tag="\([^"]*\)".*/\1/'`
        total=`echo $line | sed 's/.*value="\([^"]*\)".*/\1/'`
        iterations=`echo $line | sed 's/.*iterations="\([^"]*\)".*/\1/'`

        value=`echo "scale=4; $total/$iterations" | bc`
        plugin=`echo $tag | sed 's/.*plugin=\[\([^\[]*\)\].*/\1/'`
        matrixSize=`echo $tag | sed 's/.*matrixSize=\[\([^\[]*\)\].*/\1/'`

        filename=`echo $plugin.data | sed 's/\s/_/g'`

        if [ -z "${currentOperationsPerPlugin[$plugin]}" ]
        then
                echo -e "#matrixSize\tvalue" > $filename
                currentOperationsPerPlugin[$plugin]="$matrixSize"
        fi
        echo -e "$matrixSize\t$value" >> $filename

    done
}

echo
echo "*** Benchmarking pipeline"
echo

echo "Running benchmark ..."
./pipelinebenchmark -xml testFourierPropagation > testdata.xml

echo "Parsing data ..."
parseTestData

#set blas lib back to to netlib (if openblas is loaded acml has problems with multithreading)
sudo update-alternatives --set libblas.so.3gf /usr/lib/libblas/libblas.so.3gf

echo "
        set terminal pngcairo size 700,400
        set output 'fourierPropagation.png'
        set grid
        set key right bottom
        set logscale y
        set xlabel 'Matrixgröße' font 'Times,13'
        set ylabel 'Berechnungszeit [ms]' font 'Times,13'
        plot    'CBlas_Math_Plugin.data' using 1:2 with lines linecolor rgb 'blue' title 'CBlas(ATLAS)', \
                'GSL_Math_Plugin.data' using 1:2 with lines linecolor rgb '#FF00FF' title 'GSL', \
                'Cuda_Math_Plugin.data' using 1:2 with lines linecolor rgb '#006400' title 'Cuda', \
                'Cuda_Math_Plugin_(Single_Precision).data' using 1:2 with lines linecolor rgb 'greenyellow' title 'Cuda SP', \
                'AMD_ACML_Math_Plugin.data' using 1:2 with lines linecolor rgb 'black' title 'ACML', \
                '`ls AMD_APPML_Math_Plugin*.data`' using 1:2 with lines linecolor rgb 'coral' title 'APPML'
" | gnuplot

