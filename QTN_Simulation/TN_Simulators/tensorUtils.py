import numpy as np

def svQiskitStyleToMine(sv):
    # Qiskit: last..3rd,2nd,1st
    # Mine: 1st2nd3rd...last
    n = int(np.log2(len(sv)))
    newSv = [0 for i in range(len(sv))]
    for s in range(len(sv)):
        b = bin(s)
        str = b[2:]
        for i in range(n-len(str)):
            str = "0" + str
        newS = str[::-1]
        newInt = eval('0b' + newS)
        newSv[s] = sv[newInt]
    # print(newSv)
    # print(sv)
    return np.array(newSv)