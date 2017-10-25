# Tensorflow/Machine Learning testing
The goal of this exercise is to use Python and some Machine Learning libraries (such as SKLearn and Google's Tensorflow) to assess performance of CPU versus GPU for certain test iterations.
Ultimately, this will serve as a helpful primer in the realm of Machine Learning, such that we can determine advantages or disadvantages to solution architectures, certain types of hardware, or ML libraries.

## Baseline Test: Array Calculation
The reason we are performing a baseline array calculation is to look at how well Tensorflow performs a matrix multiplication.
Usung numpy, we used the same matrix sizes and a random number generator for both the CPU-enabled and GPU-enabled TF engines.
With Tensorflow, the GPU-enabled engine is used by default for all tasks, unless you were to manually turn it off in the configuration file, environment variables, or thye code itself.
In order to call all items in the same program, we used a command to *only* use the CPU:
```
with tf.device("/cpu:0"):
```
For smaller matrices, the CPU-enabled Tensorflow engine (using dual Xeon E5-2670s) was the clear winner. 
However, Tensorflow's GPU 

|Matrix 1 A |	Matrix 1 B |	Size |	Matrix 2 A |	Matrix 2 B |	Size |	CPU Solution (ms)	| GPU Solution (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | 
| 30 	| 40 |	 1,200 | 	 40 | 	 60 | 	 2,400 | 	3 |	12|
| 300 	| 400 	| 120,000 |	 400 | 	 600 | 	 240,000 | 	3 |	16 |
| 3,000 |	 4,000 	| 12,000,000 	| 4,000 |	 6,000 	| 24,000,000 |	36 |	58 |
| 30,000 |	 40,000 |	 1,200,000,000 | 	 40,000 	| 60,000 | 	 2,400,000,000 | 	4,745	| 5,496 |
