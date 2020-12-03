#!/bin/bash
# @ job_name		= heatomp.extrae
# @ partition		= debug
# @ initialdir		= .
# @ output		= heatomp.extrae.%j.out
# @ error		= heatomp.extrae.%j.err
# @ total_tasks		= 1
# @ cpus_per_task	= 12
# @ wall_clock_limit	= 00:02:00

executable=./heatomp
n_threads=8

module load extrae

export EXTRAE_CONFIG_FILE=extrae.xml
export OMP_NUM_THREADS=$n_threads

export LD_PRELOAD=${EXTRAE_HOME}/lib/libomptrace.so

${executable} test.dat
