
*Task 1*

---

# **2D Convolution — The Dhurandhar Way** 

## ---

This assignment focuses on optimizing the performance of 2D convolution using various optimization techniques. You will implement and analyze multiple methods, measure their impact on performance, and gain a deeper understanding of the underlying principles that drive performance improvements.

**Note**: Please refrain from modifying any part of the code other than the 2D convolutions function.

| List of Tasks (6 points) |  |  |
| ----- | :---- | :---: |
| 1A | Loop unrolling and reordering | 1 point |
| 1B | Tiling | 1 point |
| 1C | SIMD | 2 points |
| 1D | Sabka saath sabka vikaas | 2 points |

## 

## **Task 1A: Loop unrolling and reordering**



Analyze the provided 2D convolution code and implement loop reordering and unrolling optimizations to improve execution time.

By the end of this task, you should be able to answer the following questions:

1\. What motivated you to make changes to your code?  
2\. What considerations did you take into account when implementing those changes?  
3\. How effective are your changes? (Which metric will you use to quantify effectiveness and why?)

---

## **Task 1B: Tiling**



#### **1\.  *Profiling baseline 2D convolution code:***

* Find out the size of the **L1 Data (L1-D) cache** on your system.  
* Profile the baseline (naive) 2D convolution implementation and report the **L1-D cache misses per kilo instructions (MPKI)**. 

  *Hint: use a tool called perf*

#### **2\.  *Implementing Tiled 2D convolution***

* Implement a **tiled** version of 2D convolution.  
* Experiment with different **tile sizes**.  
* For each tile size, measure the **L1-D cache MPKI** to evaluate cache behavior.  
* Identify the **tile size** that gives the best performance on your hardware in terms of cache efficiency and execution time.

#### 

#### **3\.  *Collecting and analyzing different metrics of interest:***

* Test your tiled implementation on matrices of various sizes.  
* Calculate the **speedup** achieved by the tiled version compared to the baseline implementation for each matrix size.  
* Create plots to visualize the performance trends:  
  * **L1-D MPKI vs. Matrix Size** for different tile sizes.  
  * **Speedup vs. Matrix Size** for different tile sizes.

  * 

Answer the following questions:

1. Report the changes in **L1-D MPKI** observed when moving from naive to tiled 2D convolution. Justify your observations.  
2. How did **L1-D MPKI** vary across different matrix sizes and tile sizes? Explain your findings in terms of the cache hierarchy and working set sizes.  
3. Did you achieve a **speedup**? If yes, quantify the improvement and identify the contributing factors. If not, analyze the limiting factors and propose possible solutions.

---

## 

## **Task 1C: SIMD**

![][image3]  
In this task, you will utilize SIMD (Single Instruction, Multiple Data) instructions to vectorize 2D convolution and analyze their impact on instruction count and overall performance.

#### **1\.  *Baseline Profiling:***

* Report the **number of instructions** executed for the **naive** 2D convolution implementation.

#### **2\.  *SIMD Implementation:***

* Modify your 2D convolution to leverage **SIMD instructions**.  
* Implement versions using **128-bit**, **256-bit**, and (if available) **512-bit** SIMD registers.

#### **3\.  *Instruction Count and Performance Analysis:***

* Report the **number of instructions** executed when running only the **SIMD version**.  
* **Compare performance** between the naive and SIMD implementations by calculating the **speedup** achieved.

**4\.  *Multi-Size Evaluation:***

* Run your implementations on matrices of various sizes.  
* Analyze how performance and speedup vary with **matrix size** and **SIMD register width**.

#### **5\.  *Visualization:***

* Create plots showing the **speedup** for different matrix sizes and SIMD widths.  
* Choose matrix sizes that allow you to **clearly observe trends and draw meaningful conclusions**.

Answer the following:

1. Report the change in the number of instructions you observed when moving from the naive to the SIMD implementation. Justify your observations.  
2. Did you achieve any speedup? If so, how much, and what contributed to it? If not, what were the reasons?  
3. Which SIMD intrinsics did you use? Justify your choice of functions.

## 

## 

## **Task 1D: Sabka saath sabka vikaas**

In this part of the assignment, you are expected to demonstrate how combining multiple optimization techniques, such as tiling, SIMD vectorization, loop unrolling, and cache-aware programming, can lead to better performance improvements than applying each technique in isolation.

**Final performance summary for 2D convolution:**

Implement a version of 2D convolution that combines two or more of the previously explored optimization techniques.

Demonstrate how these techniques complement each other to provide synergistic performance gains, i.e., the combined benefit is greater than the sum of individual optimizations.

Include a plot that compares the best performance achieved by each optimization technique:


Additionally, plot the speedup of each optimized version (`conv_reorder`,
`conv_unroll`, `conv_tile`, `conv_simd`, `conv_optimized`) over `conv_naive`,
varying both the input image dimensions and the kernel size. Submit a report
(PDF) that includes these plots and justifies the observed performance
gains or losses for each optimization technique.