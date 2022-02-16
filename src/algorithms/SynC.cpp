//
// Created by mrjak on 14-07-2021.
//

#include <cmath>
#include <ctime>
#include <iostream>

#include "SynC.h"
#include "../utils/CPU_math.h"
#include "../structure/RTree.h"

neighborhood ComputeNeighborhood(int p, float eps, float *D, int n, int d) {
    neighborhood N;

    float *x = &D[p * d];
    for (int j = 0; j < n; j++) {
        float *y = &D[j * d];
        float dist = distance(x, y, d);
        if (dist <= eps)
            N.push_back(j);
    }

    return N;
}

void UpdatePoint(int x, const neighborhood &N_p, float *D_current, float *D_next, int n, int d) {
    for (int l = 0; l < d; l++) {
        D_next[x * d + l] = D_current[x * d + l];

        float sum = 0.;
        for (int i: N_p) {
            sum += sin(D_current[i * d + l] - D_current[x * d + l]);
        }
        D_next[x * d + l] += sum / N_p.size();
    }
}

float ComputeLocationOrder(int x, const neighborhood &N_x, float *D, int n, int d) {
    if (N_x.size() == 0)
        return 1.;

    float sum = 0.;
    for (int i: N_x) {
        sum += exp(-abs(distance(&D[x * d], &D[i * d], d)));
    }
    //    if (x == 0) {
    //        printf("CPU: r_c=%f/%d=%f\n", sum, N_x.size(), sum / N_x.size());
    //    }
    float r_c = sum / N_x.size();

    return r_c;
}

clustering synCluster(float *D, int n, int d, float eps) {
    int *C = full(n, -1);
    clustering Cs;

    int cl = 0;

    for (int i = 0; i < n; i++) {
        if (C[i] == -1) {
            bool in_cluster = false;
            cluster C_cl;
            C_cl.push_back(i);
            for (int j = 0; j < n; j++) { // we can do this better
                if (i != j && distance(&D[i * d], &D[j * d], d) <= eps) { //todo dist not needed!
                    C[j] = cl;
                    C_cl.push_back(j);
                    in_cluster = true;
                }
            }
            if (in_cluster) {
                C[i] = cl;
                Cs.push_back(C_cl);
                cl += 1;
            }
        }
    }

    delete C;
    return Cs;
}

outliers Outliers(float *D, int n, int d, clustering &C) {
    outliers O;
    for (int i = 0; i < n; i++) {
        for (cluster C_j: C) {
            if (!contains(C_j, i)) {
                O.push_back(i);
            }
        }
    }
    return O;
}

float NN(float *D, int n, int d, int k) {

    float *dists = full(k, INF);

    float avg = 0;

    for (int i = 0; i < n; i++) {
        float *x = &D[i * d];
        set_all(dists, k, INF);
        for (int j = 0; j < n; j++) {
            if (i == j)
                continue;
            float *y = &D[j * d];
            float dist = distance(x, y, d);
            if (dist < dists[0]) {
                int l = 1;
                while (l < k && dist < dists[l]) {
                    dists[l - 1] = dists[l];
                    l += 1;
                }

                dists[l - 1] = dist;
            }
        }
        avg += dists[0]; //mean(dists, k);
    }
    return avg / n;
}

float K(float x) {
    return pow(2 * PI, -1. / 2.) * exp(-pow(x, 2) / 2.);
}

float f_hat(int x, float *D, int n, int d, float *h) {
    float sum = 0.;
    for (int y = 0; y < n; y++) {
        float prod = 1.;
        for (int i = 0; i < d; i++) {
            prod *= 1. / h[i] * K((D[x * d + i] - D[y * d + i]) / h[i]);
        }
        sum += prod;
    }
    return 1. / n * sum;
}

float L(float *D, const int n, const int d, clustering &M) {

    clustering C = M;

    float L_M = 0.;
    for (int i = 0; i < C.size(); i++) {
        L_M += C[i].size() * log2(((float) n) / ((float) C[i].size()));
        L_M += d / 2. * log2((float) C[i].size());
    }

    //    printf("Got to: line %d in file %s\n", __LINE__, __FILE__);

    const float *sigma = variance(D, n, d);
    const float *IQR = iqr(D, n, d);
    float *h = new float[d];
    for (int j = 0; j < d; j++) {
        h[j] = 0.9 * pow(n, (-1 / (d + 4))) * min(sigma[j], IQR[j] / 1.34);
    }
    float sum_f_hat = 0;
    for (int y = 0; y < d; y++) {
        sum_f_hat += f_hat(y, D, n, d, h);
    }

    float L_D_given_M = 0;
    for (int i = 0; i < C.size(); i++) {
        for (auto x: C[i]) {
            float pdf_x = f_hat(x, D, n, d, h) / sum_f_hat;
            L_D_given_M += std::log2f(pdf_x);
        }
    }
    L_D_given_M *= -1;

    delete sigma;
    delete IQR;
    delete h;

    return L_M + L_D_given_M;
}

float L_Clust(float *D, const int n, const int d, clustering &M) {

    clustering C = M;

    float L_M = 0.;
    for (int i = 0; i < C.size(); i++) {
        L_M += C[i].size() * log2(((float) n) / ((float) C[i].size()));
        L_M += d / 2. * log2((float) C[i].size());
    }

    //    printf("Got to: line %d in file %s\n", __LINE__, __FILE__);

    const float *sigma = variance(D, n, d);
    const float *IQR = iqr(D, n, d);
    float *h = new float[d];
    for (int j = 0; j < d; j++) {
        h[j] = 0.9 * pow(n, (-1 / (d + 4))) * min(sigma[j], IQR[j] / 1.34);
    }

    float L_D_given_M = 0;
    for (int i = 0; i < C.size(); i++) {

        float sum_f_hat = 0;
        for (auto y: C[i]) {
            sum_f_hat += f_hat(y, D, n, d, h);
        }

        for (auto x: C[i]) {
            float pdf_x = f_hat(x, D, n, d, h) / sum_f_hat;
            L_D_given_M += std::log2f(pdf_x);
        }
    }
    L_D_given_M *= -1;

    delete sigma;
    delete IQR;
    delete h;

    return L_M + L_D_given_M;
}

clustering DynamicalClustering(float *D, int n, int d, float eps, float lam) {
    float *D_current = clone(D, n * d);
    float *D_next = clone(D, n * d);

    float r_local = 0.;
    while (r_local < lam) {
        r_local = 0.;
        for (int p = 0; p < n; p++) {
            const neighborhood N_t = ComputeNeighborhood(p, eps, D_current, n, d);
            UpdatePoint(p, N_t, D_current, D_next, n, d);
//            const neighborhood N_tp = ComputeNeighborhood(p, eps, D_current, n, d);
            float r_p = ComputeLocationOrder(p, N_t, D_current, n, d);
            r_local += r_p;
        }
        r_local /= n;
        printf("r_local: %f\n", r_local);

        float *tmp = D_next;
        D_next = D_current;
        D_current = tmp;
    }

    return synCluster(D_current, n, d, eps);
}

clustering SynC(float *h_D, int n, int d, int k) {
    int l = 0;
    float eps = NN(h_D, n, d, k);
    float delta_eps = NN(h_D, n, d, k + 1) - eps;
    bool global_synchronization = false;
    std::vector <clustering> M;
    std::vector<float> coding_cost;
    while (!global_synchronization) {
        clustering C = DynamicalClustering(h_D, n, d, eps, 1 - 1e-3);
        M.push_back(C);
        if (M[l].size() == 1) {
            global_synchronization = true;
        }
        float cost = L(h_D, n, d, M[l]);
        coding_cost.push_back(cost);
        eps += delta_eps;
        l += 1;
    }

    int l_star = arg_min(coding_cost); // forall l
    clustering M_star = M[l_star];
    return M_star;
}


clustering FDynamicalClustering(float *D, int n, int d, float eps, float lam, int B) {
    float *D_current = clone(D, n * d);
    float *D_next = clone(D, n * d);


    RTree tree(d, B);
    std::clock_t start;

    start = std::clock();
    for (int p = 0; p < n; p++) {
        tree.insert(new std::pair<int, float *>(p, &D_current[p * d]));
    }
    std::cout << "Time: " << (std::clock() - start) / (double) (CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

    float r_local = 0.;
    while (r_local < lam) {
        r_local = 0.;

        start = std::clock();
        for (int p = 0; p < n; p++) {
            const neighborhood N_t = tree.range(eps, &D_current[p * d]);//ComputeNeighborhood(p, eps, D_current, n, d);
            UpdatePoint(p, N_t, D_current, D_next, n, d);
            float r_p = ComputeLocationOrder(p, N_t, D_current, n, d);
            r_local += r_p;
        }
        std::cout << "Time: " << (std::clock() - start) / (double) (CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

        start = std::clock();
        for (int p = 0; p < n; p++) {

            //delete point
            tree.remove(new std::pair<int, float *>(p, &D_current[p * d]));

//            if (!tree.check()) {
//                printf("Line %d in file %s\n", __LINE__, __FILE__);
//                throw std::invalid_argument("invalid MBR");
//            }

            //insert point
            tree.insert(new std::pair<int, float *>(p, &D_next[p * d]));
        }
        std::cout << "Time: " << (std::clock() - start) / (double) (CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

        r_local /= n;
        printf("r_local: %f\n", r_local);

        float *tmp = D_next;
        D_next = D_current;
        D_current = tmp;
    }

    return synCluster(D_current, n, d, eps);
}