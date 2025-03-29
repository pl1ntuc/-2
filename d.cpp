#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
#include <random>
#include <cblas.h>
#include <Windows.h>
#include <omp.h>
#include <immintrin.h>

using namespace std;

constexpr int N = 4096;
using Complex = complex<double>;

void generate_matrix(vector<Complex>& matrix) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(-1.0, 1.0);
    for (auto& elem : matrix) {
        elem = Complex(dis(gen), dis(gen));
    }
}

void printProgressBar(int percent, int barWidth = 50) {
    int pos = barWidth * percent / 100;
    cout << "\r[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos)
            cout << "=";
        else if (i == pos)
            cout << ">";
        else
            cout << " ";
    }
    cout << "] " << percent << "%";
    cout.flush();
}

void naive_multiplication(const vector<Complex>& A, const vector<Complex>& B, vector<Complex>& C) {
    int lastPercent = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            Complex sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
        int currentPercent = (i + 1) * 100 / N;
        if (currentPercent > lastPercent) {
            lastPercent = currentPercent;
            printProgressBar(currentPercent);
        }
    }
    cout << endl;
}


void optimized_multiplication_avx(const vector<Complex>& A, const vector<Complex>& B, vector<Complex>& C) {
    constexpr int block_size = 64;
#pragma omp parallel for schedule(dynamic)
    for (int bi = 0; bi < N; bi += block_size) {
        for (int bk = 0; bk < N; bk += block_size) {
            for (int bj = 0; bj < N; bj += block_size) {
                int i_max = min(bi + block_size, N);
                int k_max = min(bk + block_size, N);
                int j_max = min(bj + block_size, N);
                for (int i = bi; i < i_max; i++) {
                    for (int k = bk; k < k_max; k++) {
                        double a_real = A[i * N + k].real();
                        double a_imag = A[i * N + k].imag();
                        __m256d a_re = _mm256_set1_pd(a_real);
                        __m256d a_im = _mm256_set1_pd(a_imag);
                        for (int j = bj; j < j_max; j += 2) {
                            __m256d b_val = _mm256_loadu_pd(reinterpret_cast<const double*>(&B[k * N + j]));
                            __m256d b_shuffled = _mm256_permute_pd(b_val, 0x5);
                            __m256d prod = _mm256_addsub_pd(_mm256_mul_pd(a_re, b_val), _mm256_mul_pd(a_im, b_shuffled));
                            __m256d c_val = _mm256_loadu_pd(reinterpret_cast<double*>(&C[i * N + j]));
                            c_val = _mm256_add_pd(c_val, prod);
                            _mm256_storeu_pd(reinterpret_cast<double*>(&C[i * N + j]), c_val);
                        }
                    }
                }
            }
        }
    }
}

void blas_multiplication(const vector<Complex>& A, const vector<Complex>& B, vector<Complex>& C) {
    const Complex alpha(1.0, 0.0);
    const Complex beta(0.0, 0.0);
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N,
        &alpha, A.data(), N, B.data(), N, &beta, C.data(), N);
}

void zero_matrix(vector<Complex>& C) {
    fill(C.begin(), C.end(), Complex(0.0, 0.0));
}

template <typename Func>
double measure_time(Func func) {
    auto start = chrono::high_resolution_clock::now();
    func();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

int main() {
    SetConsoleOutputCP(1251);
    SetConsoleCP(1251);

    vector<Complex> A(N * N), B(N * N), C(N * N);
    generate_matrix(A);
    generate_matrix(B);

    zero_matrix(C);
    double time_naive = measure_time([&]() { naive_multiplication(A, B, C); });
    cout << "Наивное умножение: " << time_naive << " секунд\n";

    zero_matrix(C);
    double time_blas = measure_time([&]() { blas_multiplication(A, B, C); });
    cout << "BLAS умножение: " << time_blas << " секунд\n";

    zero_matrix(C);
    double time_optimized = measure_time([&]() { optimized_multiplication_avx(A, B, C); });
    cout << "Оптимизированное умножение (AVX + OpenMP): " << time_optimized << " секунд\n";

    double computations = 2.0 * N * N * N;
    double mflops_naive = computations / (time_naive * 1e6);
    double mflops_blas = computations / (time_blas * 1e6);
    double mflops_optimized = computations / (time_optimized * 1e6);

    cout << "Производительность (MFLOPS):\n";
    cout << "Наивное: " << mflops_naive << "\n";
    cout << "BLAS: " << mflops_blas << "\n";
    cout << "Оптимизированное: " << mflops_optimized << "\n";

    cout << " Изотов Никита Антонович , ПОВа-о24 090301 ";

    cin.get();

    return 0;
}