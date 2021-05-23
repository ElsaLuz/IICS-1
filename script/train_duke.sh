# https://www.geeksforgeeks.org/pythonpath-environment-variable-in-python/ 
export PYTHONPATH=$PYTHONPATH:./  #for test_duke.sh only command-line arguments are changed. 
python ./example/iics.py --dataset dukemtmc --checkpoint path_to_checkpoint #1.It is used to set the path for the user-defined modules so 
                                                                            #that it can be directly imported into a Python program.
                                                                            #2. Command-line arguments being passed to the iics.py script 
                                                                            #that is being executed by the sh ./script/train_duke.sh shell script
