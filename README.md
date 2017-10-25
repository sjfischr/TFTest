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
However, Tensorflow's GPU began to catch up to the CPU as the matrices grew in size.
*Note:* We could not go further in this experiment as we ran out of memory on the CPU, and TensorFlow does not permit sending it any Tensors larger than 2GB. 

|Matrix 1 A |	Matrix 1 B |	Size |	Matrix 2 A |	Matrix 2 B |	Size |	CPU Solution (ms)	| GPU Solution (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | 
| 30 	| 40 |	 1,200 | 	 40 | 	 60 | 	 2,400 | 	3 |	12|
| 300 	| 400 	| 120,000 |	 400 | 	 600 | 	 240,000 | 	3 |	16 |
| 3,000 |	 4,000 	| 12,000,000 	| 4,000 |	 6,000 	| 24,000,000 |	36 |	58 |
| 30,000 |	 40,000 |	 1,200,000,000 | 	 40,000 	| 60,000 | 	 2,400,000,000 | 	4,745	| 5,496 |

### Outputs from Baseline Test
```
name: GeForce GTX 1050 Ti
major: 6 minor: 1 memoryClockRate (GHz) 1.4805
pciBusID 0000:81:00.0
Total memory: 4.00GiB
Free memory: 3.31GiB
2017-10-25 11:13:09.614570: I c:\tf_jenkins\home\workspace\release-win\m\windows-gpu\py\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:961] DMA: 0
2017-10-25 11:13:09.614767: I c:\tf_jenkins\home\workspace\release-win\m\windows-gpu\py\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:971] 0:   Y
2017-10-25 11:13:09.614994: I c:\tf_jenkins\home\workspace\release-win\m\windows-gpu\py\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:81:00.0)
2017-10-25 11:13:09.818710: I c:\tf_jenkins\home\workspace\release-win\m\windows-gpu\py\36\tensorflow\core\common_runtime\gpu\gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 1050 Ti, pci bus id: 0000:81:00.0)
Total GPU Runtime Duration was 12 milliseconds.
[[  87.31034422   56.61480665  102.03525254   87.07123712   87.84025146
    96.37660512]
 [  53.76936654  109.93654622  151.98626457  114.38538603   25.32093881
   156.81795348]
 [  79.92754731   77.59366257  109.81844484   63.61068571   50.98241952
   110.44947724]]
Intel64 Family 6 Model 45 Stepping 6, GenuineIntel
Total CPU Runtime Duration was 3 milliseconds.
[[  87.31034422   56.61480665  102.03525254   87.07123712   87.84025146
    96.37660512]
 [  53.76936654  109.93654622  151.98626457  114.38538603   25.32093881
   156.81795348]
 [  79.92754731   77.59366257  109.81844484   63.61068571   50.98241952
   110.44947724]]
   ```
   
