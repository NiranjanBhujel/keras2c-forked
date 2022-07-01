"""
File name:      keras2c_mod.py.py
Written by:     Niranjan Bhujel
Date:           June 31, 2022
Description:    Contains modification to original keras2c.
"""


import os
import zipfile
import numpy as np
from .keras2c_main import k2c
from keras2c.io_parsing import get_model_io_names
from keras2c.weights2c import Weights2C
import math
import subprocess


def __generate_call__(model, function_name, filename, malloc, verbose):
    input_shape = []
    model_inputs, model_outputs = get_model_io_names(model)
    num_inputs = len(model_inputs)
    num_outputs = len(model_outputs)

    stack_vars, malloc_vars, static_vars = Weights2C(model, function_name, malloc).write_weights(verbose)

    stateful = len(static_vars) > 0

    for i in range(num_inputs):
        temp_input_shape = np.array(model.inputs[i].shape)
        temp_input_shape = np.where(
            temp_input_shape == None, 1, temp_input_shape)
        if stateful:
            temp_input_shape = temp_input_shape[:]
        else:
            temp_input_shape = temp_input_shape[1:]
        input_shape.insert(i, temp_input_shape)
    rand_inputs = []
    for j, _ in enumerate(model_inputs):
        rand_input = 4*np.random.random(input_shape[j]) - 2
        if not stateful:
            rand_input = rand_input[np.newaxis, ...]
        rand_inputs.insert(j, rand_input)
        # make predictions
    outputs_val = model.predict(rand_inputs)

    num_inputs = len(model_inputs)
    num_outputs = len(model_outputs)  

    inputs = [f'input{k+1}' for k in range(num_inputs)]
    outputs = [f'output{k+1}' for k in range(num_outputs)]

    inputs_float = [f'float *input{k+1}' for k in range(num_inputs)]
    outputs_float = [f'float *output{k+1}' for k in range(num_outputs)]

    s = f'#include "k2c_include.h"\n'
    s += f'#include "{function_name}.h"\n'
    s += f'#include "{filename}.h"\n\n'

    s += f'void {filename}({", ".join(inputs_float)}, {", ".join(outputs_float)})\n'
    s += "{\n"

    for k in range(num_inputs):
        s += "k2c_tensor " + inputs[k] + "_tensor = {\n"
        s += f"{inputs[k]},\n"
        s += f"{len(input_shape[k])},\n"
        s += f"{math.prod(input_shape[k])},\n"
        tmp = [str(i) for i in input_shape[k]]
        for k in range(5-len(tmp)):
            tmp.append('1')
        s += "{" + ', '.join(tmp) + "}};\n\n"


    for k in range(num_outputs):
        output_shape = outputs_val[k].shape
        s += "k2c_tensor " + outputs[k] + "_tensor = {\n"
        s += f"{outputs[k]},\n"
        s += f"{len(output_shape)},\n"
        s += f"{math.prod(output_shape)},\n"
        tmp = [str(i) for i in output_shape]
        for k in range(5-len(tmp)):
            tmp.append('1')
        s += "{" + ', '.join(tmp) + "}};\n\n"
    
    call_args = ["&" + inputs[k] + "_tensor" for k in range(num_inputs)]
    call_args += ["&" + outputs[k] + "_tensor" for k in range(num_outputs)]
    s += f"{function_name}({', '.join(call_args)});\n"
    s += "}"

    with open(filename+ ".c", 'w') as fw:
        fw.write(s)

    s = f'#include "{function_name}.h"\n#include "k2c_include.h"\n\n'
    s += f'void {filename}({", ".join(inputs_float)}, {", ".join(outputs_float)});\n'
    with open(filename+ ".h", 'w') as fw:
        fw.write(s)

    try:
        subprocess.run(['astyle', '-n', filename + '.h'])
        subprocess.run(['astyle', '-n', filename + '.c'])
    except FileNotFoundError:
        print("astyle not found, {} and {} will not be auto-formatted".format(filename + ".h", filename + ".c"))


def generate(model, function_name, same_dir=True, malloc=False, num_tests=0, verbose=True):
    """
    Generate C code for specified keras model.

    Parameters
    ----------
    model : keras.model
        Keras model for which C code is requried.
    function_name : str
        Desired name of the function and file
    same_dir : bool, optional
        Whether all required C files are to be added in current directory. If True, all .c and .h files are added in current directory. If False, they are added in 'include' folder on current directory, by default True
    malloc : bool, optional
        Whether memory allocation is required, by default False
    num_tests : int, optional
        Number of tests to perform to validate the generated code, by default 0
    verbose : bool, optional
        Whether info are to be displayed, by default True
    Examples
    --------

    >>> import tensorflow as tf
    >>> from tensorflow import keras
    >>> import keras2c
    >>> model = keras.models.Sequential()
    >>> model.add(keras.layers.Flatten(input_shape=(3, )))
    >>> model.add(keras.layers.Dense(5, activation="relu"))
    >>> model.add(keras.layers.Dense(5, activation="relu"))
    >>> model.add(keras.layers.Dense(1, activation="tanh"))
    >>> keras2c.generate(model, 'model_file', same_dir=False, malloc=False, num_tests=0)
    All checks passed
    Gathering Weights
    Writing layer  flatten
    Writing layer  dense
    Writing layer  dense_1
    Writing layer  dense_2
    Formatted  model_file.h
    Formatted  model_file.c
    Done
    C code is in 'model_file.c' with header file 'model_file.h'
    A subdirectory or file include already exists.
    Unchanged  model_file_call.h
    Formatted  model_file_call.c
    Build command:
    gcc -std=c99 -I./include/ include/*.c model_file.c model_file_call.c <OTHER_C_FILE> -o <EXE_NAME>
    
    """    
    lib_files = [
        'k2c_activations.c',
        'k2c_convolution_layers.c',
        'k2c_core_layers.c',
        'k2c_embedding_layers.c',
        'k2c_helper_functions.c',
        'k2c_merge_layers.c',
        'k2c_normalization_layers.c',
        'k2c_pooling_layers.c',
        'k2c_recurrent_layers.c',
    ]
    try:
        os.remove(f'{function_name}.c')
    except:
        pass
    try:
        os.remove(f'{function_name}.h')
    except:
        pass

    k2c(model, function_name, malloc, num_tests, verbose)

    if same_dir:
        with open(function_name + '.c', 'r') as fw:
            z = fw.read()
        z = z.replace('./include/', '')
        with open(function_name + '.c', 'w') as fw:
            fw.write(z)

        with open(function_name + '.h', 'r') as fw:
            z = fw.read()
        z = z.replace('./include/', '')
        with open(function_name + '.h', 'w') as fw:
            fw.write(z)

        zip_path = f"{os.path.dirname(os.path.realpath(__file__))}\\data\\include.zip"
        dir_dest = ""

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dir_dest)

    else:
        zip_path = f"{os.path.dirname(os.path.realpath(__file__))}\\data\\include.zip"

        if 'include' not in os.listdir('/'):
            os.system('mkdir include')
        dir_dest = "include"

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dir_dest)

    __generate_call__(model, function_name, function_name + '_call', malloc, verbose)

    if verbose:
        print("Build command:", end='\n')
        if same_dir:
            # print(f"gcc -std=c99 -o <executable_name> {function_name}.c <OTHER_C_FILE> -L./ -l:libkeras2c.a -lm")
            print(f"gcc -std=c99 {' '.join(lib_files)} {function_name}.c {function_name}_call.c <OTHER_C_FILE> -o <EXE_NAME>")
        else:
            tmp = [f'include/{x}' for x in lib_files]
            print(f"gcc -std=c99 -I./include/ include/*.c {function_name}.c {function_name}_call.c <OTHER_C_FILE> -o <EXE_NAME>")