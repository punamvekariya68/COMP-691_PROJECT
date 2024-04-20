Project README

Description:
The project is an analysis of the Optimal algorithm, LRU algorithm, BlindOracle algorithm, and Combined algorithm.
The goal is to analyze the trends of the page faults of the Optimal algorithm, LRU algorithm, Blind Oracle algorithm,
and Combined algorithm based on input values like cache size, noise addition which depends on noise parameter tau and
noise parameter w.

The project consists of the following files:
1. main.py: The main file that contains all algorithms and plot graphs for all the algorithms.
2. readme.txt: A file that contains the description of the project.
3. Report.pdf: A file that contains the analysis of the Graphs.

Compilation and Execution:

To compile and execute the code:

1. Compilation Command:
   - Use the following command in the terminal to run the code:
     python main.py

2. Interpreter Version:
   - The code is written in Python 3.12.0.

3. Expected Output:
   - The code will generate total eight graphs for the Optimal algorithm, LRU algorithm, Blind Oracle algorithm, and Combined algorithm.
    - The graphs represent four trends and two regimes for each trend.
    - The trends are:
        - Average number of page faults vs. cache size.
        - Average number of page faults vs. noise parameter w.
        - Average number of page faults vs. locality parameter epsilon.
        - Average number of page faults vs. noise parameter tau.
    - The regimes are:
        - Regime 1 is where OPT is performing better than Blind Oracle which is performing better than LRU.
        - Regime 2 is where OPT is performing better than LRU which is performing better than Blind Oracle.

Code contains the following functions and their validations:

1. generateRandomSequence(k, N, n, epsilon):
   - Validated that k is less than or equal to both N and n.
   - Validated that epsilon is within the range [0, 1].
   - Validated that all parameters (k, N, and n) are greater than 0.

2. generateH(seq):
   - Validated that the input sequence is not empty.

3. addNoise(h_sequence, tau, w):
   - Validated that tau is within the range [0, 1].
   - Validated w is greater than 0.
   - Validated that the H-predicted sequence is not empty.

4. blindOracle(k, seq, hseq):
   - Validated that the input sequences seq and hseq have the same length.
   - Validated that k is greater than 0.
   - Validated that the input sequence seq is not empty.
   - Validated that k is less than or equal to the length of seq.

5. LRU(seq, k):
    - Validated that the input sequence seq is not empty.
    - Validated that k is greater than 0.
    - Validated that k is less than or equal to the length of seq.

6. combinedAlg(seq, k, hseq, tau, w):
    - Validated that the input sequence seq is not empty.
    - Validated that k is greater than 0.
    - Validated that the H-predicted sequence is not empty.
    - Validated that tau is within the range [0, 1].

