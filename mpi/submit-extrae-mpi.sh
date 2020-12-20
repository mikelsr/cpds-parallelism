#!/bin/bash
	# @ job_name		= heatmpi.extrae
	# @ partition		= debug
	# @ initialdir		= .
	# @ output		= heatmpi.extrae.%j.out
	# @ error		= heatmpi.extrae.%j.err
	# @ total_tasks		= 4
	# @ cpus_per_task	= 1
	# @ tasks_per_node	= 4
	# @ wall_clock_limit	= 00:02:00

prog=heatmpi

module load extrae

srun ./trace.sh ./$prog test.dat

