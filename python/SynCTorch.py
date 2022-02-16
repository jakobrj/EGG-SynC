import torch
import scipy
import numpy as np


def ComputeNeighborhood(p, eps, D):
    N = D.shape[0]

    neighborhood = []

    x = D[p]
    for j in range(N):
        y = D[j]
        dist = torch.norm(x - y)
        if dist < eps:
            neighborhood.append(j)

    return neighborhood


def ComputeKNN(p, k, D):
    N = D.shape[0]

    distances = torch.full(k, double('inf'))
    neighborhood = torch.zeros(k)

    x = D[p]
    for j in range(N):
        if j == p:
            continue
        y = D[j]
        dist = torch.norm(x - y)
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
        D_next[x] = D_current[x] + 1 / len(N_p) * torch.sum(torch.sin(D_current[N_p] - D_current[x]), dim=0)
    else:
        D_next[x] = D_current[x]


def ComputeLocationOrder(x, N_x, D):
    N = D.shape[0]

    if len(N_x) == 0:
        return 1

    r_c = 1 / len(N_x) * torch.sum(torch.exp(-torch.abs(torch.norm(D[x] - D[N_x], dim=1))))

    return r_c


def synCluster(D):
    N = D.shape[0]

    C = torch.full((N,), -1)
    Cs = []

    cl = 0

    for i in range(N):
        if C[i] == -1:
            in_cluster = False
            C_cl = []
            for j in range(N):
                if i != j and torch.allclose(D[i], D[j]):
                    C[j] = cl
                    C_cl.append(j)
                    in_cluster = True
            if in_cluster:
                C[i] = cl
                C_cl.append(i)
                cl += 1
                Cs.append(C_cl)

    return Cs

    # R = []
    # for C in Cs:
    #     if len(C) > 3:
    #         R.append(C)
    #
    # return R


def Outliers(D, Cl):
    N = D.shape[0]

    O = []
    for i in range(N):
        not_in_cluster = False
        for C_j in Cl:
            if i in C_j:
                not_in_cluster = True
        if not_in_cluster:
            O.append(i)

    return O


def NN(k, D):
    N = D.shape[0]
    d = D.shape[1]

    distances = torch.full((k,), double('inf'))

    avg = 0

    for i in range(N):
        x = D[i]
        distances[:] = double('inf')
        for j in range(N):
            if j == i:
                continue
            y = D[j]
            dist = torch.norm(x - y)
            if dist < distances[0]:
                l = 1
                while l < k and dist < distances[l]:
                    distances[l - 1] = distances[l]
                    l += 1

                distances[l - 1] = dist
        avg += distances[0]  # torch.mean(distances)

    return avg / N


def K(x):
    return (2 * np.pi) ** (-1 / 2) * torch.exp(-x ** 2 / 2)


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
    #     L_M += 1 * torch.log2(N)
    #     L_M += p_i / 2 * torch.log2(1)

    sigma = [torch.var(D[:, j]) for j in range(d)]
    IQR = [scipy.stats.iqr(D[:, j]) for j in range(d)]
    h = [0.9 * (N ** (-1 / (d + 4))) * min(sigma[j], IQR[j] / 1.34) for j in range(d)]
    sum_f_hat = torch.sum(torch.DoubleTensor([f_hat(y, D, h) for y in range(N)]))

    L_D_given_M = 0
    for i in range(K):
        # sigma = torch.array([torch.var(D[C[i], j]) for j in range(d)])
        # IQR = torch.array([scipy.stats.iqr(D[C[i], j]) for j in range(d)])
        # h = torch.array([0.9 * (N ** (-1 / (d + 4))) * min(sigma[j], IQR[j] / 1.34) for j in range(d)])
        # sum_f_hat = torch.sum([f_hat_in_C(y, D, h, C[i]) for y in C[i]])
        # sum_f_hat = torch.sum([f_hat(y, D[C[i],:], h, ) for y in range(len(C[i]))])
        for x in C[i]:
            pdf_x = f_hat(x, D, h) / sum_f_hat
            # pdf_x = f_hat_in_C(x, D, h, C[i]) / sum_f_hat

            L_D_given_M += torch.log2(pdf_x)

    # for x in O:
    #     pdf_x = f_hat(x, D, h) / sum_f_hat
    #     L_D_given_M += torch.log2(pdf_x)

    L_D_given_M *= -1
    print("L_M:", L_M)
    print("L_D_given_M:", L_D_given_M)
    return L_M + L_D_given_M


def L_Clust(D, M):  # todo something seems to be wrong with the cost function???
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
    #     L_M += 1 * torch.log2(N)
    #     L_M += p_i / 2 * torch.log2(1)

    sigma = [torch.var(D[:, j]) for j in range(d)]
    IQR = [scipy.stats.iqr(D[:, j]) for j in range(d)]
    h = [0.9 * (N ** (-1 / (d + 4))) * min(sigma[j], IQR[j] / 1.34) for j in range(d)]
    # sum_f_hat = torch.sum(torch.DoubleTensor([f_hat(y, D, h) for y in range(N)]))

    L_D_given_M = 0
    for i in range(K):
        # sigma = torch.FloatTensor([torch.var(D[C[i], j]) for j in range(d)])
        # IQR = torch.FloatTensor([scipy.stats.iqr(D[C[i], j]) for j in range(d)])
        # h = torch.FloatTensor([0.9 * (len(C[i]) ** (-1 / (d + 4))) * min(sigma[j], IQR[j] / 1.34) for j in range(d)])
        # sum_f_hat = torch.sum([f_hat_in_C(y, D, h, C[i]) for y in C[i]])
        sum_f_hat = torch.sum(torch.DoubleTensor([f_hat(y, D[C[i], :], h, ) for y in range(len(C[i]))]))
        for x in range(len(C[i])):
            # pdf_x = f_hat(x, D, h) / sum_f_hat
            # pdf_x = f_hat_in_C(x, D, h, C[i]) / sum_f_hat
            pdf_x = f_hat(x, D[C[i], :], h) / sum_f_hat

            L_D_given_M += torch.log2(pdf_x)

    # for x in O:
    #     pdf_x = f_hat(x, D, h) / sum_f_hat
    #     L_D_given_M += torch.log2(pdf_x)

    L_D_given_M *= -1
    print("L_M:", L_M)
    print("L_D_given_M:", L_D_given_M)
    return L_M + L_D_given_M


def L_Clust_Outl(D, M):  # todo something seems to be wrong with the cost function???
    N = D.shape[0]
    d = D.shape[1]
    C = M[0]
    O = M[1]
    K = len(C)
    p_i = d

    L_M = 0
    outliers = N
    for i in range(K):
        outliers -= len(C[i])
        L_M += len(C[i]) * np.log2(N / len(C[i]))
        L_M += p_i / 2 * np.log2(len(C[i]))

    L_M += outliers * np.log2(N)

    # for x in range(len(O)):
    #     L_M += 1 * torch.log2(N)
    #     L_M += p_i / 2 * torch.log2(1)

    sigma = [torch.var(D[:, j]) for j in range(d)]
    IQR = [scipy.stats.iqr(D[:, j]) for j in range(d)]
    h = [0.9 * (N ** (-1 / (d + 4))) * min(sigma[j], IQR[j] / 1.34) for j in range(d)]
    # sum_f_hat = torch.sum(torch.FloatTensor([f_hat(y, D, h) for y in range(N)]))

    L_D_given_M = 0
    for i in range(K):
        # sigma = torch.FloatTensor([torch.var(D[C[i], j]) for j in range(d)])
        # IQR = torch.FloatTensor([scipy.stats.iqr(D[C[i], j]) for j in range(d)])
        # h = torch.FloatTensor([0.9 * (len(C[i]) ** (-1 / (d + 4))) * min(sigma[j], IQR[j] / 1.34) for j in range(d)])
        # sum_f_hat = torch.sum([f_hat_in_C(y, D, h, C[i]) for y in C[i]])
        sum_f_hat = torch.sum(torch.DoubleTensor([f_hat(y, D[C[i], :], h, ) for y in range(len(C[i]))]))
        for x in range(len(C[i])):
            # pdf_x = f_hat(x, D, h) / sum_f_hat
            # pdf_x = f_hat_in_C(x, D, h, C[i]) / sum_f_hat
            pdf_x = f_hat(x, D[C[i], :], h) / sum_f_hat

            L_D_given_M += torch.log2(pdf_x)

    L_D_given_M += outliers * ((2 * np.pi) ** (-1. / 2.)) * np.prod(h)

    # for x in O:
    #     pdf_x = f_hat(x, D, h) / sum_f_hat
    #     L_D_given_M += torch.log2(pdf_x)

    L_D_given_M *= -1
    print("L_M:", L_M)
    print("L_D_given_M:", L_D_given_M)
    return L_M + L_D_given_M


def DynamicalClustering(D, eps, lam=1 - 1e-3, call=None):
    D_current = D.clone()
    D_next = D.clone()
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

        # print("r_local:", r_local)
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
        print("l:", l, "eps:", eps)
        M.append(DynamicalClustering(D, eps, call=call))
        if len(M[l][0]) == 1:
            global_synchronization = True
        print("number of clusters:", len(M[l][0]))
        print("computing cost...")
        coding_cost.append(L_Clust_Outl(D, M[l]))
        print("cost:", coding_cost[-1])
        eps += delta_eps
        l += 1

    l_star = torch.argmin(torch.DoubleTensor(coding_cost))  # forall l
    print("best:", l_star, coding_cost[l_star])
    print("number of clusters:", len(M[l_star][0]))
    M_star = M[l_star]
    return M_star[0]
