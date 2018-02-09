# S2S_sum

Apply sequence-to-sequence model to generate summary text according to given text


```
CUDA_VISIBLE_DEVICES=0 python main_nt.py --options set_values_for_options
```

Set progress to any available GPU.

## Required packages:
* Python 2.7
* Tensorflow 1.0
* Tensorgraph


## Note: 
The rouge calculation in this file uses PythonRouge (https://github.com/tagucci/pythonrouge).

I will update the calculation of PyRouge (https://pypi.python.org/pypi/pyrouge/0.1.0) later.

