# https://www.geeksforgeeks.org/pythonpath-environment-variable-in-python/ 
export PYTHONPATH=$PYTHONPATH:./  #for test_duke.sh only argparse flags are changed. Here setting the environment path along with argparse flags to call it like: sh ./script/train_market.sh
python ./example/iics.py --dataset dukemtmc --checkpoint path_to_checkpoint #It is used to set the path for the user-defined modules so 
                                                                            #that it can be directly imported into a Python program.
