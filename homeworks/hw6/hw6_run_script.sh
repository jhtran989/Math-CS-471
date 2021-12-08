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

parallel_error_dir="parallel_error/"
parallel_error_script="hw6_parallel_error.pbs"
parallel_error_python="hw6_parallel_final.py"

# changed from run_script_inner_prod
function run_parallel_error_script() {
	#move module load to the job script as well
	#module load mpich-3.2-gcc-4.8.5-7ebkszx
	
	mkdir -p ${parallel_error_dir}
	cd ${parallel_error_dir}
	
	# copy shell script to current dir
	cp ${up}${parallel_error_script} ${parallel_error_script}
	
	# copy python script to current dir
	cp ${up}${parallel_error_python} ${parallel_error_python}
	cp ${up}${poisson_python} ${poisson_python}
	
	qsub ${parallel_error_script}
}

function run_script_matvec_2d {
	module load mpich-3.2-gcc-4.8.5-7ebkszx
	
	if (( ${normal} == 1 )); then 
		mpirun -n ${num_processes} ./example
	fi 
	
	if (( ${tau_trace} == 1 )); then 
		export TAU_TRACE=1
	fi
	
	if (( ${tau_profile} == 1 )); then 
		mpirun -n ${num_processes} tau_exec ./example
	fi
}

function run_script_cannon {
	module load mpich-3.2-gcc-4.8.5-7ebkszx
	
	if (( ${normal} == 1 )); then 
		mpirun -n ${num_processes} ./example
	fi 
	
	if (( ${tau_trace} == 1 )); then 
		export TAU_TRACE=1
	fi
	
	if (( ${tau_profile} == 1 )); then 
		mpirun -n ${num_processes} tau_exec ./example
	fi
}

# module load mpich-3.2-gcc-4.8.5-7ebkszx
# mpirun -n ${num_processes} ./example

# ;; vs exit;; (one continues and the other exits)

# OPTIONS
while getopts "p:" args; do
	case $args in 
	h)
		echo "============================================================================================="
		echo "Arguments:"
		echo "  -p	[np]	where np is the number of processes"
		echo "============================================================================================="
		exit;;
	n)
		normal=1
		choice="${OPTARG}"
		echo "============================================================================================="
		echo "Executing normally"
		echo "============================================================================================="
		run_script
		exit;;
	p)
		choice=${OPTARG}
		echo "============================================================================================="
		echo "Executing parallel algorithm to create the solution/error plot"
		echo "============================================================================================="
		run_parallel_error_script
		exit;;
	t)
		tau_profile=1
		tau_trace=1
		choice=${OPTARG}
		echo "============================================================================================="
		echo "Executing with Tau tracing (and profiling)"
		echo "============================================================================================="
		run_script
		exit;;
	\?)
		echo "============================================================================================="
		echo "Sorry, invalid option. Please use option -h for valid options."
		echo "============================================================================================="
		exit;;
	esac
done 