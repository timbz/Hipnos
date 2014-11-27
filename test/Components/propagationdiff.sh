#!/bin/bash
echo "
        set term pngcairo size 900,300
        set output 'propagation-error.png'
        set xlabel 'Propagationsstrecke [m]' font 'Times,13'
        set ylabel 'Intensitätsdifferenz' font 'Times,13'
        set grid
        plot    'propagation-error.data' using 1:2 with lines title ''
        pause -1;

" | gnuplot

echo "
        set term pngcairo size 900,300
        set output 'propagation-with-lens-error.png'
        set xlabel 'Propagationsstrecke [m]' font 'Times,13'
        set ylabel 'Intensitätsdifferenz' font 'Times,13'
        set grid
        plot    'propagation-with-lens-error.data' using 1:2 with lines title ''
        pause -1;

" | gnuplot
