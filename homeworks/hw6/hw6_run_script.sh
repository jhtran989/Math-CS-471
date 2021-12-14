# processes array
num_processes=4
process_array=("1" "4" "16")

# program names
programs_root="dense_linear_alg/"

inner_prod_program="inner_prod"
matvec_program="matvec"
matvec_2d_program="matvec_2d"
cannon_program="cannon"

# dir names
up="../"
up2=${up}${up}
up3=${up}${up2}
root="/users/jtran989/cs491/hw4/dense-linear-algebra-Astrolace/"${programs_root}

inner_prod_output="inner_prod_output/"
matvec_output="matvec_output/"
matvec_2d_output="matvec_2d_output/"
cannon_output="cannon_output/"

profile_dir="profile"
trace_dir="trace"

# flags for Tau profiling and tracing (and normal execution)
normal=0
tau_profile=0
tau_trace=0

# command line argument for the programs
num_elements_one=1000
num_elements_ten=10000

poisson_python="poisson.py"

# Error plot to test convergence -- REMEMBER TO UPDATE BELOW
serial_error_dir="serial_error/"
serial_error_script="hw6_serial_error.pbs"
serial_error_python="hw6_serial_error.py"

# Error plot to test convergence -- REMEMBER TO UPDATE BELOW
parallel_error_dir="parallel_error/"
parallel_error_script="hw6_parallel_error.pbs"
parallel_error_python="hw6_parallel_error.py"

# Record timings for weak scaling -- REMEMBER TO UPDATE BELOW
parallel_weak_dir="parallel_weak/"
parallel_weak_script="hw6_parallel_weak.pbs"
parallel_weak_python="hw6_parallel_weak.py"

# Record timings for weak scaling -- REMEMBER TO UPDATE BELOW
# now run multiiple weak scaling studies at once with different tolerances to compare timings...

parallel_weak_dir_array=("parallel_weak_100_1e-4/")
parallel_weak_script="hw6_parallel_weak.pbs"

# parallel_strong_python_array=("hw6_parallel_strong_256_1e-4.py" "hw6_parallel_strong_256_1e-8.py" "hw6_parallel_strong_512_1e-4.py" "hw6_parallel_strong_512_1e-8.py")
parallel_weak_python_array=()
for parallel_weak_dir in ${parallel_weak_dir_array[@]}
do
	parallel_weak_python_array+=("hw6_${parallel_weak_dir%/}.py")
done
num_weak=${#parallel_weak_dir_array[@]}

# Record timings for strong scaling -- REMEMBER TO UPDATE BELOW
# now run multiiple strong scaling studies at once with different tolerances to compare timings...
parallel_strong_dir="parallel_strong/"
parallel_strong_script="hw6_parallel_strong.pbs"
parallel_strong_python="hw6_parallel_strong.py"

# Record timings for strong scaling -- REMEMBER TO UPDATE BELOW
# now run multiiple strong scaling studies at once with different tolerances to compare timings...

#TODO: uncomment
# parallel_strong_dir_array=("parallel_strong_256_1e-4/" "parallel_strong_256_1e-8/" "parallel_strong_512_1e-4/"
# "parallel_strong_512_1e-8/")
# parallel_strong_script="hw6_parallel_strong.pbs"
#
# # parallel_strong_python_array=("hw6_parallel_strong_256_1e-4.py" "hw6_parallel_strong_256_1e-8.py" "hw6_parallel_strong_512_1e-4.py" "hw6_parallel_strong_512_1e-8.py")
# parallel_strong_python_array=()
# for parallel_strong_dir in ${parallel_strong_dir_array[@]}
# do
# 	parallel_strong_python_array+=("hw6_${parallel_strong_dir%/}.py")
# done
# num_strong=${#parallel_strong_dir_array[@]}

# for Michael
num_strong=1
parallel_strong_dir_array=("parallel_strong_128_1e-4/")
parallel_strong_script="hw6_parallel_strong.pbs"
parallel_strong_python_array=("hw6_parallel_strong_128_1e-4.py")

# changed from run_script_inner_prod
function run_serial_error_script() {
	#move module load to the job script as well
	#module load mpich-3.2-gcc-4.8.5-7ebkszx
	
	# clean out directory to remove any artifacts from past debug/run session
	[[ -d ${serial_error_dir} ]] && rm -r ${serial_error_dir}
	
	mkdir -p ${serial_error_dir}
	cd ${serial_error_dir}
	
	# copy shell script to current dir
	cp ${up}${serial_error_script} ${serial_error_script}
	
	# copy python script to current dir
	cp ${up}${serial_error_python} ${serial_error_python}
	cp ${up}${poisson_python} ${poisson_python}
	
	# escape '$' with '\' since there are some variables only seen in the .pbs script
	sed -i "81 i python ${serial_error_python}" ${serial_error_script}
	sed -i "82 i tracejob \$PBS_JOBID" ${serial_error_script}
	
	qsub ${serial_error_script}
}

# changed from run_script_inner_prod
function run_parallel_error_script() {
	#move module load to the job script as well
	#module load mpich-3.2-gcc-4.8.5-7ebkszx
	
	# clean out directory to remove any artifacts from past debug/run session
	[[ -d ${parallel_error_dir} ]] && rm -r ${parallel_error_dir}
	
	mkdir -p ${parallel_error_dir}
	cd ${parallel_error_dir}
	
	# copy shell script to current dir
	cp ${up}${parallel_error_script} ${parallel_error_script}
	
	# copy python script to current dir
	cp ${up}${parallel_error_python} ${parallel_error_python}
	cp ${up}${poisson_python} ${poisson_python}
	
	# escape '$' with '\' since there are some variables only seen in the .pbs script
	sed -i "81 i mpirun -machinefile \$PBS_NODEFILE -np ${num_processes} --map-by node:PE=8 python ${parallel_error_python}" ${parallel_error_script}
	sed -i "82 i tracejob \$PBS_JOBID" ${parallel_error_script}
	
	qsub ${parallel_error_script}
}

function run_parallel_weak_script() {
	#move module load to the job script as well
	#module load mpich-3.2-gcc-4.8.5-7ebkszx
	
	# make directory for strong scaling
	mkdir -p ${parallel_weak_dir}
	cd ${parallel_weak_dir}
	
	# clean out directory to remove any artifacts from past debug/run session
	[[ -d ${num_processes} ]] && rm -r ${num_processes}
	
	# want to keep a copy of the output file by the PBS system for each run, so need separate directories (on different number of processes)
	mkdir -p ${num_processes}
	cd ${num_processes}
	
	# copy shell script to current dir
	cp ${up2}${parallel_weak_script} ${parallel_weak_script}
	
	# copy python script to current dir
	cp ${up2}${parallel_weak_python} ${parallel_weak_python}
	cp ${up2}${poisson_python} ${poisson_python}
	
	# calculate the corresponding number of nodes and processes
	num_nodes=$(( num_processes / 8 ))
	remainder=$(( num_processes % 8 ))

	# just round up if there are still some processes left over
	if (( ${remainder} != 0 )); then
		num_nodes=$(( num_nodes + 1 ))
	fi

	num_processes_per_node=8
	
	# escape '$' with '\' since there are some variables only seen in the .pbs script
	sed -i "9 i #PBS -lnodes=${num_nodes}:ppn=${num_processes_per_node}" ${parallel_weak_script}
	sed -i "82 i mpirun -machinefile \$PBS_NODEFILE -np ${num_processes} --map-by node:PE=8 python ${parallel_weak_python}" ${parallel_weak_script}
	sed -i "83 i tracejob \$PBS_JOBID" ${parallel_weak_script}
	
	qsub ${parallel_weak_script}
}

function run_parallel_strong_script() {
	#move module load to the job script as well
	#module load mpich-3.2-gcc-4.8.5-7ebkszx
	
	# make directory for strong scaling
	mkdir -p ${parallel_strong_dir}
	cd ${parallel_strong_dir}
	
	# clean out directory to remove any artifacts from past debug/run session
	[[ -d ${num_processes} ]] && rm -r ${num_processes}
	
	# want to keep a copy of the output file by the PBS system for each run, so need separate directories (on different number of processes)
	mkdir -p ${num_processes}
	cd ${num_processes}
	
	# copy shell script to current dir
	cp ${up2}${parallel_strong_script} ${parallel_strong_script}
	
	# copy python script to current dir
	cp ${up2}${parallel_strong_python} ${parallel_strong_python}
	cp ${up2}${poisson_python} ${poisson_python}
	
	# calculate the corresponding number of nodes and processes
	num_nodes=$(( num_processes / 8 ))
	remainder=$(( num_processes % 8 ))

	# just round up if there are still some processes left over
	if (( ${remainder} != 0 )); then
		num_nodes=$(( num_nodes + 1 ))
	fi

	num_processes_per_node=8
	
	# increase time a lot for final results...
	# ah, just use the full 48 hours, just in case for 8, 4, and 2 processes
	# actually, just do it for all cases (included 16, 32, 64 processes)
	if (( $num_processes > 8 )); then
		num_hours="48"
		num_minutes="00"
	elif (( $num_processes == 8 )); then
		num_hours="48"
		num_minutes="00"
	elif (( $num_processes == 4 )); then
		num_hours="48"
		num_minutes="00"
	else # assumed to be 2 processes...
		num_hours="48"
		num_minutes="00"
	fi
	
	# if (( $num_processes > 8 )); then
# 		num_hours="02"
# 		num_minutes="30"
# 	elif (( $num_processes == 8 )); then
# 		num_hours="03"
# 		num_minutes="45"
# 	elif (( $num_processes == 4 )); then
# 		num_hours="05"
# 		num_minutes="45"
# 	else # assumed to be 2 processes...
# 		num_hours="07"
# 		num_minutes="35"
# 	fi
	
	# if (( $num_processes > 8 )); then
# 		num_hours="00"
# 		num_minutes="45"
# 	elif (( $num_processes == 8 )); then
# 		num_hours="01"
# 		num_minutes="15"
# 	elif (( $num_processes == 4 )); then
# 		num_hours="01"
# 		num_minutes="45"
# 	else # assumed to be 2 processes...
# 		num_hours="02"
# 		num_minutes="35"
# 	fi
	
	# escape '$' with '\' since there are some variables only seen in the .pbs script
	sed -i "10 i #PBS -lnodes=${num_nodes}:ppn=${num_processes_per_node}" ${parallel_strong_script}
	sed -i "16 i #PBS -l walltime=${num_hours}:${num_minutes}:00" ${parallel_strong_script}
	sed -i "86 i mpirun -machinefile \$PBS_NODEFILE -np ${num_processes} --map-by node:PE=8 python ${parallel_strong_python}" ${parallel_strong_script}
	sed -i "87 i tracejob \$PBS_JOBID" ${parallel_strong_script}
	
	qsub ${parallel_strong_script}
}

function run_multiple_parallel_strong_script() {
	#move module load to the job script as well
	#module load mpich-3.2-gcc-4.8.5-7ebkszx
	
	for (( i=0; i<${num_strong}; i++ ))
	do
		current_parallel_strong_dir=${parallel_strong_dir_array[$i]}
		current_parallel_strong_python=${parallel_strong_python_array[$i]}
		
		# make directory for strong scaling
		mkdir -p ${current_parallel_strong_dir}
		cd ${current_parallel_strong_dir}
	
		# clean out directory to remove any artifacts from past debug/run session
		[[ -d ${num_processes} ]] && rm -r ${num_processes}
	
		# want to keep a copy of the output file by the PBS system for each run, so need separate directories (on different number of processes)
		mkdir -p ${num_processes}
		cd ${num_processes}
	
		# copy pbs script to current dir
		cp ${up2}${parallel_strong_script} ${parallel_strong_script}
	
		# copy python script to current dir
		cp ${up2}${current_parallel_strong_python} ${current_parallel_strong_python}
		cp ${up2}${poisson_python} ${poisson_python}
	
		# calculate the corresponding number of nodes and processes
		num_nodes=$(( num_processes / 8 ))
		remainder=$(( num_processes % 8 ))

		# just round up if there are still some processes left over
		if (( ${remainder} != 0 )); then
			num_nodes=$(( num_nodes + 1 ))
		fi

		num_processes_per_node=8
	
		# increase time a lot for final results...
		# ah, just use the full 48 hours, just in case for 8, 4, and 2 processes
		# as well as the last few cases
		if (( $num_processes > 8 )); then
			num_hours="48"
			num_minutes="00"
		elif (( $num_processes == 8 )); then
			num_hours="48"
			num_minutes="00"
		elif (( $num_processes == 4 )); then
			num_hours="48"
			num_minutes="00"
		else # assumed to be 2 processes...
			num_hours="48"
			num_minutes="00"
		fi
	
		# escape '$' with '\' since there are some variables only seen in the .pbs script
		sed -i "10 i #PBS -lnodes=${num_nodes}:ppn=${num_processes_per_node}" ${parallel_strong_script}
		sed -i "16 i #PBS -l walltime=${num_hours}:${num_minutes}:00" ${parallel_strong_script}
		
		# maybe --map-by node:PE=8 is causing an issue...
		# sed -i "86 i mpirun -machinefile \$PBS_NODEFILE -np ${num_processes} --map-by node:PE=8 python ${current_parallel_strong_python}" ${parallel_strong_script}
		sed -i "86 i mpirun -machinefile \$PBS_NODEFILE -np ${num_processes} python ${current_parallel_strong_python}" ${parallel_strong_script}
		sed -i "87 i tracejob \$PBS_JOBID" ${parallel_strong_script}
		
		qsub ${parallel_strong_script}
		
		cd ${up2}
	done
}

function run_multiple_parallel_weak_script() {
	#move module load to the job script as well
	#module load mpich-3.2-gcc-4.8.5-7ebkszx
	
	for (( i=0; i<${num_weak}; i++ ))
	do
		current_parallel_weak_dir=${parallel_weak_dir_array[$i]}
		current_parallel_weak_python=${parallel_weak_python_array[$i]}
		
		# make directory for weak scaling
		mkdir -p ${current_parallel_weak_dir}
		cd ${current_parallel_weak_dir}
	
		# clean out directory to remove any artifacts from past debug/run session
		[[ -d ${num_processes} ]] && rm -r ${num_processes}
	
		# want to keep a copy of the output file by the PBS system for each run, so need separate directories (on different number of processes)
		mkdir -p ${num_processes}
		cd ${num_processes}
	
		# copy pbs script to current dir
		cp ${up2}${parallel_weak_script} ${parallel_weak_script}
	
		# copy python script to current dir
		cp ${up2}${current_parallel_weak_python} ${current_parallel_weak_python}
		cp ${up2}${poisson_python} ${poisson_python}
	
		# calculate the corresponding number of nodes and processes
		num_nodes=$(( num_processes / 8 ))
		remainder=$(( num_processes % 8 ))

		# just round up if there are still some processes left over
		if (( ${remainder} != 0 )); then
			num_nodes=$(( num_nodes + 1 ))
		fi

		num_processes_per_node=8
	
		# increase time a lot for final results...
		# ah, just use the full 48 hours, just in case for 8, 4, and 2 processes
		# as well as the last few cases
		if (( $num_processes > 8 )); then
			num_hours="48"
			num_minutes="00"
		elif (( $num_processes == 8 )); then
			num_hours="48"
			num_minutes="00"
		elif (( $num_processes == 4 )); then
			num_hours="48"
			num_minutes="00"
		else # assumed to be 2 processes...
			num_hours="48"
			num_minutes="00"
		fi
	
		# escape '$' with '\' since there are some variables only seen in the .pbs script
		sed -i "10 i #PBS -lnodes=${num_nodes}:ppn=${num_processes_per_node}" ${parallel_weak_script}
		sed -i "16 i #PBS -l walltime=${num_hours}:${num_minutes}:00" ${parallel_weak_script}
		
		# maybe --map-by node:PE=8 is causing an issue...
		# sed -i "86 i mpirun -machinefile \$PBS_NODEFILE -np ${num_processes} --map-by node:PE=8 python ${current_parallel_weak_python}" ${parallel_weak_script}
		sed -i "86 i mpirun -machinefile \$PBS_NODEFILE -np ${num_processes} python ${current_parallel_weak_python}" ${parallel_weak_script}
		sed -i "87 i tracejob \$PBS_JOBID" ${parallel_weak_script}
		
		qsub ${parallel_weak_script}
		
		cd ${up2}
	done
}

# module load mpich-3.2-gcc-4.8.5-7ebkszx
# mpirun -n ${num_processes} ./example

# ;; vs exit;; (one continues and the other exits)

# OPTIONS
while getopts ":hap:w:s:x:y:" args; do
	case $args in 
	h)
		echo "============================================================================================="
		echo "Arguments:"
		echo "  -p	[np]	normal parallel algorithm to create the error plots"
		echo "  -w	[np]	create timings in parallel for the WEAK scaling study"
		echo "  -s	[np]	create timings in parallel for the STRONG scaling study"
		echo " where [np] is the number of processes "
		echo "============================================================================================="
		exit;;
	a)
		echo "============================================================================================="
		echo "Executing serial algorithm to create the solution/error plot"
		echo "============================================================================================="
		run_serial_error_script
		exit;;
	p)
		num_processes=${OPTARG}
		echo "============================================================================================="
		echo "Executing parallel algorithm to create the solution/error plot"
		echo "number of processes: ${num_processes}"
		echo "============================================================================================="
		run_parallel_error_script
		exit;;
	w)
		num_processes=${OPTARG}
		echo "============================================================================================="
		echo "Executing parallel algorithm to make the timings file for WEAK scaling"
		echo "number of processes: ${num_processes}"
		echo "============================================================================================="
		run_parallel_weak_script
		exit;;
	s)
		num_processes=${OPTARG}
		echo "============================================================================================="
		echo "Executing parallel algorithm to make the timings file for STRONG scaling"
		echo "number of processes: ${num_processes}"
		echo "============================================================================================="
		run_parallel_strong_script
		exit;;
	x)
		num_processes=${OPTARG}
		echo "============================================================================================="
		echo "Executing parallel algorithm to make the timings file for STRONG scaling for MULTIPLE cases"
		echo "number of processes: ${num_processes}"
		echo "============================================================================================="
		run_multiple_parallel_strong_script
		exit;;
	y)
		num_processes=${OPTARG}
		echo "============================================================================================="
		echo "Executing parallel algorithm to make the timings file for WEAK scaling for MULTIPLE cases"
		echo "number of processes: ${num_processes}"
		echo "============================================================================================="
		run_multiple_parallel_weak_script
		exit;;
	\?)
		echo "============================================================================================="
		echo "Sorry, invalid option. Please use option -h for a list of valid options."
		echo "============================================================================================="
		exit;;
	esac
done 

if [ $OPTIND -eq 1 ]; 
then 
	echo "============================================================================================="
	echo "No options were passed. Please use option -h for a list of valid options."; 
	echo "============================================================================================="
fi
	