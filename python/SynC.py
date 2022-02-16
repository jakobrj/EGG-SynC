import numpy as np
import scipy


def ComputeNeighborhood(p, eps, D):
    N = D.shape[0]

    neighborhood = []

    x = D[p]
    for j in range(N):
        y = D[j]
        dist = np.linalg.norm(x - y)
        if dist < eps:
            neighborhood.append(j)

    return np.array(neighborhood)


def ComputeKNN(p, k, D):
    N = D.shape[0]

    distances = np.full(k, np.inf)
    neighborhood = np.zeros(k)

    x = D[p]
    for j in range(N):
        if j == p:
            continue
        y = D[j]
        dist = np.linalg.norm(x - y)
        if dist < distances[0]:
            l = 1
            while l < k and dist < distances[l]:
                distances[l - 1] = distances[l]
                neighborhood[l - 1] = neighborhood[l]
                l += 1

            distances[l - 1] = dist
            neighborhood[l - 1] = j

    return neighborhood


def UpdatePoint(x, N_p, D_current, D_next):
    if len(N_p) > 0:
        D_next[x] = D_current[x] + 1 / len(N_p) * np.sum([
            np.sin(D_current[y] - D_current[x])  # todo why sin?
            # D_current[y] - D_current[x]
            for y in N_p
        ], axis=0)
    else:
        D_next[x] = D_current[x]


def ComputeLocationOrder(x, N_x, D):
    N = D.shape[0]

    if len(N_x) == 0:
        return 1

    r_c = 1 / len(N_x) * np.sum([
        np.exp(-np.abs(np.linalg.norm(D[x] - D[y])))
        for y in N_x
    ])

    return r_c


def synCluster(D):
    N = D.shape[0]

    C = np.full(N, -1)
    Cs = []

    cl = 0

    for i in range(N):
        if C[i] == -1:
            in_cluster = False
            C_cl = []
            for j in range(N):
                if i != j and np.allclose(D[i], D[j]):
                    C[j] = cl
                    C_cl.append(j)
                    in_cluster = True
            if in_cluster:
                C[i] = cl
                C_cl.append(i)
                cl += 1
                Cs.append(C_cl)

    return Cs


def Outliers(D, Cl):
    N = D.shape[0]

    O = []
    for i in range(N):
        for C_j in Cl:
            if i not in C_j:
                O.append(i)

    return np.array(O)


def NN(k, D):
    N = D.shape[0]
    d = D.shape[1]

    distances = np.full(k, np.inf)

    avg = 0

    for i in range(N):
        x = D[i]
        distances[:] = np.inf
        for j in range(N):
            if j == i:
                continue
            y = D[j]
            dist = np.linalg.norm(x - y)
            if dist < distances[0]:
                l = 1
                while l < k and dist < distances[l]:
                    distances[l - 1] = distances[l]
                    l += 1

                distances[l - 1] = dist
        avg += distances[0]#np.mean(distances)

    return avg / N


def K(x):
    return (2 * np.pi) ** (-1 / 2) * np.exp(-x ** 2 / 2)


def f_hat(x, D, h):
    N = D.shape[0]
    d = D.shape[1]

    sum = 0
    for y in range(N):
        prod = 1
        for i in range(d):
            prod *= 1 / h[i] * K((D[x, i] - D[y, i]) / h[i])
        sum += prod
    return 1 / N * sum


def f_hat_in_C(x, D, h, C):
    N = D.shape[0]
    d = D.shape[1]

    sum = 0
    for y in C:
        prod = 1
        for i in range(d):
            prod *= 1 / h[i] * K((D[x, i] - D[y, i]) / h[i])
        sum += prod
    return 1 / len(C) * sum


# def pdf(x, D, h, sum_f_hat):
#     return f_hat(x, D, h) / sum_f_hat


def L(D, M):  # todo something seems to be wrong with the cost function???
    N = D.shape[0]
    d = D.shape[1]
    C = M[0]
    O = M[1]
    K = len(C)
    p_i = d

    L_M = 0
    for i in range(K):
        L_M += len(C[i]) * np.log2(N / len(C[i]))
        L_M += p_i / 2 * np.log2(len(C[i]))

    # for x in range(len(O)):
    #     L_M += 1 * np.log2(N)
    #     L_M += p_i / 2 * np.log2(1)

    sigma = np.array([np.var(D[:, j]) for j in range(d)])
    IQR = np.array([scipy.stats.iqr(D[:, j]) for j in range(d)])
    h = np.array([0.9 * (N ** (-1 / (d + 4))) * min(sigma[j], IQR[j] / 1.34) for j in range(d)])
    sum_f_hat = np.sum([f_hat(y, D, h) for y in range(N)])

    L_D_given_M = 0
    for i in range(K):
        # sigma = np.array([np.var(D[C[i], j]) for j in range(d)])
        # IQR = np.array([scipy.stats.iqr(D[C[i], j]) for j in range(d)])
        # h = np.array([0.9 * (N ** (-1 / (d + 4))) * min(sigma[j], IQR[j] / 1.34) for j in range(d)])
        # sum_f_hat = np.sum([f_hat_in_C(y, D, h, C[i]) for y in C[i]])
        # sum_f_hat = np.sum([f_hat(y, D[C[i],:], h, ) for y in range(len(C[i]))])
        for x in C[i]:
            pdf_x = f_hat(x, D, h) / sum_f_hat
            # pdf_x = f_hat_in_C(x, D, h, C[i]) / sum_f_hat

            L_D_given_M += np.log2(pdf_x)

    # for x in O:
    #     pdf_x = f_hat(x, D, h) / sum_f_hat
    #     L_D_given_M += np.log2(pdf_x)

    L_D_given_M *= -1
    print("L_M:", L_M)
    print("L_D_given_M:", L_D_given_M)
    return L_M + L_D_given_M


def DynamicalClustering(D, eps, lam=1 - 1e-3, call=None):
    D_current = D.copy()
    D_next = D.copy()
    call(D_next)
    loopFlag = True
    while loopFlag:
        r_local = 0
        for p in range(len(D)):
            N_p = ComputeNeighborhood(p, eps, D_current)
            UpdatePoint(p, N_p, D_current, D_next)
            r_p = ComputeLocationOrder(p, N_p, D_current)
            r_local += r_p
            # print("r_p:", r_p)

        r_local /= len(D)

        print("r_local:", r_local)
        if call is not None:
            call(D_next)

        if r_local > lam:  # todo this why of terminating is strange!
            loopFlag = False
            Cl = synCluster(D_next)
            Ol = Outliers(D_next, Cl)
            Ml = (Cl, Ol)
        tmp = D_next
        D_next = D_current
        D_current = tmp

    return Ml


def Sync(D, k, call=None):
    l = 0
    eps = NN(k, D)
    delta_eps = NN(k + 1, D) - eps
    global_synchronization = False
    M = []
    coding_cost = []
    while not global_synchronization:
        print("l:", l, "eps", eps)
        M.append(DynamicalClustering(D, eps, call=call))
        if len(M[l][0]) == 1:
            global_synchronization = True
        print("computing cost:")
        coding_cost.append(L(D, M[l]))
        print(coding_cost[-1])
        eps += delta_eps
        l += 1

    l_star = np.argmin(coding_cost)  # forall l
    M_star = M[l_star]
    return M_star
