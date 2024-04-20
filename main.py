import random

from matplotlib import pyplot as plt


# This function generates a random sequence of pages
def generateRandomSequence(k, N, n, epsilon):
    if k >= N or k > n:
        print("k should be less than N")
        exit()
    elif epsilon < 0 or epsilon > 1:
        print("epsilon should be in [0, 1]")
        exit()
    elif k < 1 or N < 1 or n < 1:
        print("k, N, and n should be greater than 0")
        exit()

    sequence = []
    L = set(range(1, k + 1))

    for i in range(n):
        if i < k:
            page = i + 1
        else:
            if random.random() < epsilon:
                page = random.choice(list(L))
            else:
                page = random.choice(list(set(range(1, N + 1)) - L))
                L.remove(random.choice(list(L)))
                L.add(page)
        sequence.append(page)

    return sequence


# This function generates a sequence of h values
def generateH(seq):
    n = len(seq)
    if n == 0:
        print("Sequence should not be empty")
        exit()

    h_seq = [n + 1] * n
    last_occurrence = {}
    for i in reversed(range(n)):
        page = seq[i]
        if page in last_occurrence:
            h_seq[i] = last_occurrence[page] + 1
        last_occurrence[page] = i
    return h_seq


# This function adds noise to the sequence of h values
def addNoise(hseq, tau, w):
    if tau < 0 or tau > 1:
        print("tau should be in [0, 1]")
        exit()
    elif w < 1:
        print("w should be greater than 0")
        exit()
    elif len(hseq) == 0:
        print("hseq should not be empty")
        exit()

    predicted_hseq = []
    for i in hseq:
        if random.random() < 1 - tau:
            predicted_hseq.append(i)
        else:
            l = max(i + 1, i - w // 2)
            h_predicted = random.randint(l, l + w)
            predicted_hseq.append(h_predicted)

    return predicted_hseq


# This function implements the BlindOracle algorithm
def blindOracle(k, seq, hseq):
    if len(seq) != len(hseq):
        print("seq and hseq should have the same length")
        exit()
    elif k < 1:
        print("k should be greater than 0")
        exit()
    elif len(seq) == 0:
        print("seq should not be empty")
        exit()
    elif k > len(seq):
        print("k should be less than or equal to the length of seq")
        exit()

    cache = {}
    page_faults = 0

    for i in range(len(seq)):
        if seq[i] not in cache:
            page_faults += 1
            if len(cache) == k:
                page_to_evict = max(cache, key=cache.get)
                del cache[page_to_evict]
            cache[seq[i]] = hseq[i]
        else:
            cache[seq[i]] = hseq[i]

    return page_faults


# This function implements the LRU algorithm
def LRU(k, seq):
    if k <= 0:
        print("Cache size (k) should be greater than 0.")
        exit()
    if len(seq) == 0:
        print("Sequence should not be empty")
        exit()
    cache = []
    page_faults = 0
    for page in seq:
        if page in cache:
            cache.remove(page)
            cache.append(page)
        else:
            page_faults += 1
            if len(cache) == k:
                cache.pop(0)
            cache.append(page)

    return page_faults


# This function implements the combination of LRU and BlindOracle algorithms
def combinedAlg(k, seq, hPredicted, thr):
    if len(seq) != len(hPredicted):
        print("seq and hseq should have the same length")
        exit()
    elif k < 1:
        print("k should be greater than 0")
        exit()
    elif len(seq) == 0:
        print("seq should not be empty")
        exit()
    elif thr < 0 or thr > 1:
        print("thr should be in [0, 1]")
        exit()

    f1, f2 = 0, 0
    total_faults, f1_temp_faults, f2_temp_faults = 0, 0, 0
    lru_cache = []
    blindoracle_cache = {}
    flag = "LRU"

    for i in range(len(seq)):
        if seq[i] in lru_cache:
            lru_cache.remove(seq[i])
            lru_cache.append(seq[i])
        else:
            f1 += 1
            if len(lru_cache) == k:
                lru_cache.pop(0)
            lru_cache.append(seq[i])

        if seq[i] not in blindoracle_cache:
            f2 += 1
            if len(blindoracle_cache) == k:
                page_to_evict = max(blindoracle_cache, key=blindoracle_cache.get)
                del blindoracle_cache[page_to_evict]
            blindoracle_cache[seq[i]] = hPredicted[i]
        else:
            blindoracle_cache[seq[i]] = hPredicted[i]

        if f1 > (1 + thr) * f2 and flag == "LRU":
            total_faults = total_faults + f1 + k - f1_temp_faults
            f1_temp_faults = f1
            f1 = f2
            lru_cache = []
            for j in blindoracle_cache.keys():
                lru_cache.append(j)
            flag = "blindoracle"
        elif f2 > (1 + thr) * f1 and flag == "blindoracle":
            total_faults = total_faults + f2 + k - f2_temp_faults
            f2_temp_faults = f2
            f2 = f1
            blindoracle_cache = {page: hPredicted[i] for page in lru_cache}
            flag = "LRU"
        if i == len(seq) - 1:
            if flag == "LRU":
                total_faults = total_faults + f1 - f1_temp_faults
            else:
                total_faults = total_faults + f2 - f2_temp_faults

    if total_faults == 0:
        total_faults = f1 if f1 < f2 else f2

    return total_faults


# This function runs all the tests
def run_tests():
    print("Running tests...")
    try:
        test_generateRandomSequence()
        test_generateH()
        test_addNoise()
        test_blindOracle()
        test_lru()
        test_combinedAlg()
        test_all()
        print("All tests passed successfully!")
    except Exception as e:
        print("Error occurred during testing:", str(e))


# Test functions
def test_generateRandomSequence():
    k, N, n, epsilon = 3, 10, 8, 0.6
    sequence = generateRandomSequence(k, N, n, epsilon)
    assert len(sequence) == n
    assert all(1 <= page <= N for page in sequence)
    print("generateRandomSequence test passed successfully!")


# This function tests the generateH function
def test_generateH():
    sequence = [1, 2, 3, 6, 2, 8, 2, 6]
    hseq = generateH(sequence)
    assert len(hseq) == len(sequence)
    print("generateH test passed successfully!")


# This function tests the addNoise function
def test_addNoise():
    hseq = [11, 5, 9, 8, 7, 9, 9, 9]
    tau, w = 0.2, 4
    predicted_hseq = addNoise(hseq, tau, w)
    assert len(predicted_hseq) == len(hseq)
    print("addNoise test passed successfully!")


# This function tests the blindOracle function
def test_blindOracle():
    k = 3
    seq = [1, 2, 3, 6, 2, 8, 2, 6]
    hseq = [11, 5, 9, 8, 7, 9, 9, 9]
    page_faults = blindOracle(k, seq, hseq)
    assert page_faults == 5
    k = 4
    seq = [1, 2, 3, 4, 4, 3, 1, 3, 6, 2, 6, 4, 4, 6, 5, 2, 3, 2, 7, 8]
    hseq = [9, 13, 9, 10, 16, 12, 26, 21, 14, 18, 18, 17, 23, 25, 26, 22, 23, 22, 26, 23]
    page_faults = blindOracle(k, seq, hseq)
    assert page_faults == 8
    print("blindOracle test passed successfully!")


# This function tests the LRU function
def test_lru():
    k = 4
    seq = [1, 2, 3, 4, 1, 7, 1, 3, 5, 6, 3, 4, 5, 8, 2, 8, 3, 1, 5, 7, 1, 4, 3, 3, 4, 8, 7]
    page_faults = LRU(k, seq)
    assert page_faults == 18
    k = 4
    seq = [1, 2, 3, 4, 4, 4, 2, 3, 9, 6, 4, 4, 4, 8, 7, 9, 8, 7, 7, 1]
    page_faults = LRU(k, seq)
    assert page_faults == 11
    print("LRU test passed successfully!")


# This function tests the combinedAlg function
def test_combinedAlg():
    thr = 0.4
    k = 3
    seq = [1, 2, 3, 2, 1, 5, 3, 8, 1, 1, 6, 5, 8, 5, 8, 8, 1, 5, 11, 3, 9, 10, 1, 1, 5, 6, 12, 1, 5, 12]
    hseq = [6, 8, 12, 31, 14, 14, 20, 18, 10, 17, 26, 14, 15, 18, 18, 31, 23, 28, 34, 34, 33, 31, 24, 28, 29, 31, 32,
            32, 34, 33]
    page_faults = combinedAlg(k, seq, hseq, thr)
    assert page_faults == 25
    thr = 0.3
    k = 4
    seq = [1, 2, 3, 4, 6, 1, 2, 2, 2, 4, 1, 5, 1, 5, 7, 4, 7, 2, 1, 7, 6, 4, 2, 6, 1, 4]
    hseq = [8, 8, 27, 13, 21, 11, 9, 14, 18, 20, 13, 17, 19, 28, 22, 25, 23, 23, 25, 27, 24, 26, 27, 30, 27, 29]
    page_faults = combinedAlg(k, seq, hseq, thr)
    assert page_faults == 19
    print("combinedAlg test passed successfully!")


# This function tests all the functions
def test_all():
    sequence = generateRandomSequence(k=3, N=10, n=30, epsilon=0.6)
    print("Generated sequence:", sequence)
    hseq = generateH(sequence)
    print("Generated hseq:", hseq)
    predicted_hseq = addNoise(hseq, tau=0.2, w=4)
    print("Predicted values with noise:", predicted_hseq)
    page_faults = blindOracle(3, sequence, predicted_hseq)
    print("Number of page faults with BlindOracle:", page_faults)
    page_faults = LRU(3, sequence)
    print("Number of page faults with LRU:", page_faults)
    page_faults = combinedAlg(5, sequence, hseq, 0.1)
    print("Number of page faults with combinedAlg:", page_faults)


# This function calculates the average number of page faults for each algorithm
def avgpageFaults(k, N, epsilon, tau, w):
    n = 5000
    LRUcount, BlindOraclecount, OPTcount, Combinedcount = 0, 0, 0, 0
    for i in range(100):
        seq = generateRandomSequence(k, N, n, epsilon)
        hseq = generateH(seq)
        hPredicted = addNoise(generateH(seq), tau, w)
        OPTcount += blindOracle(k, seq, hseq)
        BlindOraclecount += blindOracle(k, seq, hPredicted)
        LRUcount += LRU(k, seq)
        Combinedcount += combinedAlg(k, seq, hPredicted, 0.1)
    print("k:", k, " N:", N, " epsilon:", epsilon, " tau:", tau, " w:", w)
    print("Average number of page faults for OPT: ", OPTcount / 100)
    print("Average number of page faults for BlindOracle: ", BlindOraclecount / 100)
    print("Average number of page faults for LRU: ", LRUcount / 100)
    print("Average number of page faults for Combined: ", Combinedcount / 100)
    print()
    return [OPTcount / 100, BlindOraclecount / 100, LRUcount / 100, Combinedcount / 100]


# This function plots the graph
def plottingGraph(trend, regime, x, y):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y[0], label="OPT", marker='o')
    plt.plot(x, y[1], label="BlindOracle", marker='o')
    plt.plot(x, y[2], label="LRU", marker='o')
    plt.plot(x, y[3], label="Combined", marker='o')
    plt.xlabel(trend)
    plt.ylabel("Average number of page faults")
    plt.title(f"Trend with {trend} and Regime {regime}")
    plt.legend()
    plt.show()


# This function generates the data for the plot
def generatingDataForPlot(values, arg_function):
    OPTf, Blindf, LRUf, Combinedf = [], [], [], []
    for i in values:
        lst = arg_function(i)
        OPTf.append(lst[0])
        Blindf.append(lst[1])
        LRUf.append(lst[2])
        Combinedf.append(lst[3])
    return [OPTf, Blindf, LRUf, Combinedf]


# This function generates the data for trend 1
def trend1():
    k = [5, 10, 15, 20]
    print("Trend 1 and Regime 1")
    regime1 = generatingDataForPlot(k, lambda x: avgpageFaults(x, x * 10, 0.6, 0.7, 200))
    print("Trend 1 and Regime 2")
    regime2 = generatingDataForPlot(k, lambda x: avgpageFaults(x, x * 10, 0.6, 0.7, 1000))
    plottingGraph("k", 1, k, regime1)
    plottingGraph("k", 2, k, regime2)


# This function generates the data for trend 2
def trend2():
    w_regime1 = [100, 150, 200, 250, 300]
    w_regime2 = [1000, 1250, 1500, 1750, 2000]
    print("Trend 2 and Regime 1")
    regime1 = generatingDataForPlot(w_regime1, lambda x: avgpageFaults(10, 100, 0.6, 0.7, x))
    print("Trend 2 and Regime 2")
    regime2 = generatingDataForPlot(w_regime2, lambda x: avgpageFaults(10, 100, 0.6, 0.7, x))
    plottingGraph("w", 1, w_regime1, regime1)
    plottingGraph("w", 2, w_regime2, regime2)


# This function generates the data for trend 3
def trend3():
    epsilon = [0.4, 0.5, 0.6, 0.7]
    print("Trend 3 and Regime 1")
    regime1 = generatingDataForPlot(epsilon, lambda x: avgpageFaults(10, 100, x, 0.7, 200))
    print("Trend 3 and Regime 2")
    regime2 = generatingDataForPlot(epsilon, lambda x: avgpageFaults(10, 100, x, 0.9, 1000))
    plottingGraph("epsilon", 1, epsilon, regime1)
    plottingGraph("epsilon", 2, epsilon, regime2)


# This function generates the data for trend 4
def trend4():
    tau = [0.2, 0.4, 0.6, 0.8]
    print("Trend 4 and Regime 1")
    regime1 = generatingDataForPlot(tau, lambda x: avgpageFaults(10, 100, 0.7, x, 300))
    print("Trend 4 and Regime 2")
    regime2 = generatingDataForPlot(tau, lambda x: avgpageFaults(10, 100, 0.7, x, 2000))
    plottingGraph("tau", 1, tau, regime1)
    plottingGraph("tau", 2, tau, regime2)


# This function plots the graphs
def plot():
    trend1()
    trend2()
    trend3()
    trend4()


# This is the main function containing the menu
def main():
    while True:
        print("Choose an option:")
        print("1.Run the program")
        print("2.Run tests")
        print("3.Plots")
        choice = int(input())
        match choice:
            case 1:
                print("enter k, N, n, and epsilon to generate a random sequence(e.g. 3 10 8 0.6):")
                k, N, n, epsilon = map(float, input().split())
                sequence = generateRandomSequence(int(k), int(N), int(n), epsilon)
                print("Generated sequence:", sequence)
                hseq = generateH(sequence)
                print("Generated hseq:", hseq)
                tau, w = 0.2, 4
                predicted_hseq = addNoise(hseq, tau, w)
                print("Predicted values with noise:", predicted_hseq)
                page_faults = blindOracle(k, sequence, predicted_hseq)
                print("Number of page faults with BlindOracle:", page_faults)
                page_faults = LRU(k, sequence)
                print("Number of page faults with LRU:", page_faults)
                thr = float(input("Enter threshold value:"))
                page_faults = combinedAlg(k, sequence, hseq, thr)
                print("Number of page faults with combinedAlg:", int(page_faults))
            case 2:
                run_tests()
            case 3:
                plot()
            case _:
                print("Invalid choice!")
                exit()


if __name__ == "__main__":
    main()
