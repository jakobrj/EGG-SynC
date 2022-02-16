//
// Created by mrjak on 14-07-2021.
//

#include "CPU_math.h"

float distance(float *x, float *y, int d) {
    float dist = 0.;
    for (int l = 0; l < d; l++) {
        float diff = x[l] - y[l];
        dist += diff * diff;
    }
    return sqrt(dist);
}

float mean(float *x, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += x[i];
    }
    return sum / n;
}

float *full(int n, float value) {
    float *temp = new float[n];
    for (int i = 0; i < n; i++) {
        temp[i] = value;
    }
    return temp;
}

int *full(int n, int value) {
    int *temp = new int[n];
    for (int i = 0; i < n; i++) {
        temp[i] = value;
    }
    return temp;
}

float *full(int n, std::function<int(float)> func) {
    float *temp = new float[n];
    for (int i = 0; i < n; i++) {
        temp[i] = func(i);
    }
    return temp;
}

float *clone(float *array, int n) {
    float *temp = new float[n];
    for (int i = 0; i < n; i++) {
        temp[i] = array[i];
    }
    return temp;
}

bool contains(std::vector<int> A, int i) {
    return std::count(A.begin(), A.end(), i);
}

void set_all(float *A, int n, float value) {
    for (int i = 0; i < n; i++) {
        A[i] = value;
    }
}

float *variance(float *D, int n, int d) {
    float *sigma = new float[d];
    for (int j = 0; j < d; j++) {
        float m = 0.;
        for (int i = 0; i < n; i++) {
            m += D[i * d + j];
        }
        m /= n;

        float var = 0;
        for (int i = 0; i < n; i++) {
            var += (D[i * d + j] - m) * (D[i * d + j] - m);
        }
        var /= n;
        sigma[j] = sqrt(var);
    }

    return sigma;
}

int middle(int start, int end) {
    int n = end - start;
    int m = n / 2;
    return start + m;
}

float *iqr(float *D, int n, int d) {

    float *r = new float[d];
    float *a = new float[n];

    for (int l = 0; l < d; l++) {
        for (int i = 0; i < n; i++) {
            a[i] = D[i * d + l];
        }

        std::sort(a, a + n);

        int m2 = middle(0, n);
        int m1 = middle(0, m2 + 1); //we never include end!
        int m3 = middle(m2 + 1, n);

        float Q1 = a[m1];
        float Q3 = a[m3];
        r[l] = Q3 - Q1;
    }

    return r;
}

float min(float a, float b) {
    if (a <= b)
        return a;
    return b;
}

int arg_min(const std::vector<float> &v) {
    if (v.size() == 0)
        return 0;

    int i = 0;
    int value = v[0];
    for (int j = 1; j < v.size(); j++) {
        if (v[j] < value) {
            i = j;
            value = v[j];
        }
    }
    return i;
}

float min(const std::vector<float> &v) {
    if (v.size() == 0)
        return 0.;

    int value = v[0];
    for (int j = 1; j < v.size(); j++) {
        if (v[j] < value) {
            value = v[j];
        }
    }
    return value;
}

int maximum(int *v, int n) {
    if (n == 0)
        return 0.;

    int value = v[0];
    for (int j = 1; j < n; j++) {
        if (v[j] > value) {
            value = v[j];
        }
    }
    return value;
}

void print_array(int *x, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", x[i]);
    }
    printf("\n");
}


void print_array(int *x, int n, int m) {
    for (int i = 0; i < n * m; i++) {
        if (x[i] < 10)
            printf(" ");
        if (x[i] < 100)
            printf(" ");
        printf("%d ", x[i]);
        if ((i + 1) % m == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

void print_array(bool *x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i]) {
            printf("1 ");
        } else {
            printf("0 ");
        }
    }
    printf("\n");
}