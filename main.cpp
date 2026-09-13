#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <sched.h>
#include <time.h>

#include "Eigen/Core"
#include "Eigen/SparseCore"

extern "C" {
#include "board.h"
#include "cpu.h"
}

static void set_cpu() {
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        perror("sched_getaffinity");
        return;
    }
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed))
            continue;
        cpu_set_t selected;
        CPU_ZERO(&selected);
        CPU_SET(cpu, &selected);
        if (sched_setaffinity(0, sizeof(selected), &selected) != 0) {
            perror("sched_setaffinity");
            return;
        }
        printf("benchmark cpu = %d\n", cpu);
        return;
    }
    fprintf(stderr, "No available CPU for affinity\n");
}

static int64_t time_us() {
    struct timespec tv;
    if (clock_gettime(CLOCK_MONOTONIC, &tv) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (int64_t)tv.tv_sec * 1000000 + tv.tv_nsec / 1000;
}

static void print_eigen_info() {
    printf("Eigen %d.%d.%d, pointer bits = %zu, max align bytes = %d\n",
           EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION,
           sizeof(void*) * 8, EIGEN_MAX_ALIGN_BYTES);
    // These describe this build, not all features supported by the CPU.
    printf("Eigen compile-time SIMD:");
#ifdef EIGEN_VECTORIZE_SSE2
    printf(" SSE2");
#endif
#ifdef EIGEN_VECTORIZE_AVX
    printf(" AVX");
#endif
#ifdef EIGEN_VECTORIZE_AVX2
    printf(" AVX2");
#endif
#ifdef EIGEN_VECTORIZE_AVX512
    printf(" AVX512");
#endif
#ifdef EIGEN_VECTORIZE_NEON
    printf(" NEON");
#endif
#ifndef EIGEN_VECTORIZE
    printf(" disabled");
#endif
    printf("\n");
}

template <typename Scalar>
static void benchmark_matrix(const char *label) {
    const int size = 32;
    const int iterations = 10000;
    Eigen::Matrix<Scalar, size, size> a, b, c;
    a.setRandom();
    b.setRandom();
    c.noalias() = a * b;

    // Consume each result so the repeated products remain observable.
    volatile Scalar checksum = Scalar(0);
    int64_t start = time_us();
    for (int i = 0; i < iterations; ++i) {
        a(0, 0) = Scalar(1) + Scalar(i) / Scalar(iterations);
        c.noalias() = a * b;
        checksum += c(i % size, i % size);
    }
    int64_t elapsed = time_us() - start;
    printf("%s %dx%d iterations=%d time=%" PRId64 "us checksum=%.9g\n",
           label, size, size, iterations, elapsed, (double)checksum);

    const int dim = 132;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    Matrix d1 = Matrix::Random(dim, dim);
    Matrix d2 = Matrix::Random(dim, dim);
    start = time_us();
    Matrix d3 = d1 * d2 * d1.transpose();
    elapsed = time_us() - start;
    printf("%s ABAt %dx%d time=%" PRId64 "us checksum=%.9g\n",
           label, dim, dim, elapsed, (double)d3.sum());
}

template <typename Scalar>
static void benchmark_sparse(const char *label) {
    const int dim = 132;
    typedef Eigen::SparseMatrix<Scalar> SparseMatrix;
    SparseMatrix a(dim, dim), b(dim, dim);
    a.reserve(Eigen::VectorXi::Constant(dim, 3));
    b.reserve(Eigen::VectorXi::Constant(dim, 3));
    for (int col = 0; col < dim; ++col) {
        if (col > 0) {
            a.insert(col - 1, col) = Scalar(-0.25);
            b.insert(col - 1, col) = Scalar(0.5);
        }
        a.insert(col, col) = Scalar(2);
        b.insert(col, col) = Scalar(1);
        if (col + 1 < dim) {
            a.insert(col + 1, col) = Scalar(-0.25);
            b.insert(col + 1, col) = Scalar(0.5);
        }
    }
    a.makeCompressed();
    b.makeCompressed();

    int64_t start = time_us();
    SparseMatrix c = a * b;
    int64_t elapsed = time_us() - start;
    printf("%s sparse %dx%d time=%" PRId64 "us nnz=%ld checksum=%.9g\n",
           label, dim, dim, elapsed, (long)c.nonZeros(), (double)c.sum());
}

int main() {
    board_init();
    cpu_init();
    fields_dump(board_fields());
    fields_dump(cpu_fields());
    board_cleanup();
    cpu_cleanup();

    set_cpu();
    print_eigen_info();
    // Use the same random sequence for both scalar types.
    srand(1);
    benchmark_matrix<double>("double");
    srand(1);
    benchmark_matrix<float>("float");
    benchmark_sparse<double>("double");
    benchmark_sparse<float>("float");
    return 0;
}
