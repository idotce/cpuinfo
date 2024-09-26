#include <stdio.h>
#include <sched.h> // for sched_setaffinity()
#include <unistd.h> // for getpid()
#include <sys/types.h>
#include <sched.h>
#include <time.h>
using namespace std;

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Sparse"
using namespace Eigen;

#ifdef __cplusplus
extern "C" {
#endif
#include "board.h"
#include "cpu.h"
#ifdef __cplusplus
}
#endif

void set_cpu() {
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(0, &cpu_set);
    sched_setaffinity(getpid(), sizeof(cpu_set_t), &cpu_set);
    printf("sched_getcpu = %d\n", sched_getcpu());
}

static int64_t time_us() {
    struct timespec tv;
    clock_gettime(CLOCK_MONOTONIC, &tv); // CLOCK_REALTIME CLOCK_MONOTONIC
    return (int64_t)tv.tv_sec*1000000 + tv.tv_nsec/1000;
}

double calculate_pi(int accuracy) {
     double result = 1;
     int a = 2;
     int b = 1;
     for(int i = 0; i < accuracy; i ++){
          result = (a/(double)b) * result;
          if(a < b){
               a = a + 2;
          }
          else if(b < a){
               b = b + 2;
          }
     }
     return result * 2;
}

int main(int argc, char* argv[])
{
    rpiz_fields *bf, *pf;
    board_init();
    cpu_init();
    bf = board_fields();
    fields_dump(bf);
    pf = cpu_fields();
    fields_dump(pf);
    board_cleanup();
    cpu_cleanup();

    set_cpu();
#ifdef __CUDACC__
    printf("__CUDACC__\n");
#endif
#if (defined __ARM_NEON) || (defined __ARM_NEON__)
    printf("__ARM_NEON\n");
#endif
#if defined(__aarch64__)
  #define ARCH_ARM64 1
#else
  #define ARCH_ARM64 0
#endif
    printf("__aarch64__ = %d\n", ARCH_ARM64);
    #define align ((32+3)/4*4)
    printf("align = %d\n", align);

    Eigen::Matrix<double, align, align> a(align, align);
    Eigen::Matrix<double, align, align> b(align, align);
    Eigen::Matrix<double, align, align> c(align, align);
    Eigen::MatrixXd d(align, align);
    int64_t s1 = time_us();
    for (int i=0; i<10000; i++) {
        a.setIdentity();
        b.setIdentity();
        c = a * b;
    }
    printf("double time=%ldus\n", time_us()-s1);

    int dim = 132;
    MatrixXd d1(dim, dim);
    MatrixXd d2(dim, dim);
    d1.setIdentity();
    d2.setIdentity();
    int64_t s2 = time_us();
    MatrixXd d3 = d1*d2*d1.transpose();
    printf("double ABAt time=%ldus\n", time_us()-s2);

    return 0;
}
