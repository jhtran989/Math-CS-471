# README
## Math/CS 471
### Homework 6
#### Michael Sands, John Tran

- The homework report is in the directory
`hw6/report` and is called `homework6.pdf`.
- The code for `hw6` is in the directory `hw6/code`
- The output files are labled .output in thier respective directories  
- Since we have multiple versions of the `hw6` code, we will reference the main files to each part below
- To run the code, type the following command in the command line:

```
$ python hw6_serial_error.py
$ python hw6_parallel_error.py
$ python hw6_parallel_weak.py
$ python hw6_parallel_strong.py

```

- In addition, we have a number of `bash` scripts as well as `PBS` scripts we used to run the above `Python` code on `Wheeler`. They will also be listed below:

```
- hw6_run_script.sh // contains options that will run the above default Python scripts for each part of hw6 (individually by specifying a given option, like -s for strong scaling)
- (For more options, please check the above bash script)
- hw6_weak_scaling_script.sh // automates the submission of the weak scaling script by running through all the specified number of processes (1, 4, 16, 64)
- hw6_strong_scaling_script.sh // automates the submission of the strong scaling script by running through all the specified number of processes (2, 4, 8, 16, 32, 64)
- (Note: the two above scripts use the corresponding default Python script instead of a specified script with modified paramters, as will be case for the two bash scripts below)
- hw6_multiple_weak_scaling_script.sh // similar to the above hw6_weak_scaling_script.sh, but uses an array of given weak scaling scripts to run through all at once at the same series of processes for weak scaling (e.g., hw6_parallel_weak_100_1e-4.py)
- hw6_multiple_strong_scaling_script.sh // similar to the above hw6_strong_scaling_script.sh, but uses an array of given strong scaling scripts to run through all at once at the same series of processes for weak scaling (e.g., hw6_parallel_strong_512_1e-4.py)

```

- For additional information about the code and changing some
  parameters and output (e.g., figures), look at the comments contained in 
  the code.