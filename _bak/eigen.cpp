#include <string.h>
#include <stdlib.h>

#include <unistd.h>
#include <sys/types.h>
#include <sched.h>
#include <time.h>

using namespace std;

void set_cpu_aff()
{
	cpu_set_t my_set;
	CPU_ZERO(&my_set);
	CPU_SET(0, &my_set);
	sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
}

static int64_t getus()
{
	struct timespec tv;
	clock_gettime(CLOCK_MONOTONIC, &tv);
	return (int64_t)tv.tv_sec * 1000000 + tv.tv_nsec/1000;
}

//#define EIGEN_DONT_VECTORIZE
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <Eigen/Sparse>
using namespace Eigen;

using namespace std;

#include <intrin.h>

void test_eigen()
{
	int cpuInfo[4] = { 0, 0, 0, 0 };
    __cpuid(cpuInfo, 0);

	set_cpu_aff();
	printf("EIGEN_MAX_ALIGN_BYTES=%d\n", EIGEN_MAX_ALIGN_BYTES);

#ifdef EIGEN_DONT_VECTORIZE
	printf("DEIGEN_DONT_VECTORIZE\n");
#endif

#ifdef EIGEN_VECTORIZE_NEON
	printf("EIGEN_VECTORIZE_NEON\n");

#endif

#ifdef EIGEN_ARCH_ARM64
	printf("EIGEN_ARCH_ARM64=%d\n", EIGEN_ARCH_ARM64);
#endif

#ifdef EIGEN_ARCH_ARM32
	printf("EIGEN_ARCH_ARM32=%d\n", EIGEN_ARCH_ARM32);
#endif

	#define s 32
	#define align ((s+3)/4*4)
	#define align s

	printf("align=%d\n", align);

	#define typeD Eigen::Matrix<double, align, align>
	#define typeF Eigen::Matrix<float, align, align>

	//#define typeD Eigen::MatrixXd
	//#define typeF Eigen::MatrixXf

	std::cout << 1 << endl;
	typeD a(s,s);
	typeD b(s,s);
	typeD c(s,s);
	Eigen::MatrixXd d(s,s);
	int64_t t = getus();
	for(int i=0; i<10000; i++)
	{
		a.setIdentity();
		b.setIdentity();
		c = a * b;
	}
	//c = d;
	printf("\ndouble:%lld\n", getus() - t);

	t = getus();

	typeF fa(s,s);
	typeF fb(s,s);
	typeF fc(s,s);

	Eigen::MatrixXf fd(s,s);

	for(int i=0; i<10000; i++)
	{
		fa.setIdentity();
		fb.setIdentity();
		fc = fa * fb;
	}

	//fc=fd;
	std::cout << fc(0,0);
	printf("\nfloat:%lld\n", getus() - t);

	printf("EIGEN%d.%d.%d, %d\n", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION , EIGEN_MINOR_VERSION ,
			EIGEN_ARCH_DEFAULT_NUMBER_OF_REGISTERS);
}

using namespace Eigen;
void test_eigen2()
{
	int dim = 132;
	MatrixXd d1(dim, dim);
	MatrixXd d2(dim, dim);

	d1.setIdentity();
	d2.setIdentity();

	int64_t us = getus();
	MatrixXd d3 = d1*d2*d1.transpose();
	int dt = getus() - us;
	printf("double ABAt time=%dus\n", dt);

	MatrixXf d1f(dim, dim);
	MatrixXf d2f(dim, dim);

	d1f.setIdentity();
	d2f.setIdentity();

	us = getus();
	MatrixXf d3f = (d1f*d2f*d1f.transpose()).cast<float>();
	d1f = d1.cast<float>();
	MatrixXd d33 = d3f.cast<double>();

	SparseMatrix<float> sparse1(dim, dim);
	SparseMatrix<float> sparse2(dim, dim);
	SparseMatrix<float> sparse3(dim, dim);

	sparse1.setIdentity();
	sparse2.setIdentity();
	d3f.setIdentity();

	us = getus();
	d3f += (d1f*d2f*d1f.transpose());
	dt = getus() - us;
	printf("float ABAt time=%dus\n", dt);
	printf("%f\n", d3f(0,0));

	typedef float real;
	real v = 0;
	us = getus();
	for(int i=0; i<10000000; i++)
		v += v+1;
	dt = getus() - us;
	printf("float dt=%dus\n", dt);


	SparseMatrix<float> d5f(5, 5);
	d5f.setIdentity();
	d5f.diagonal().setConstant(0.1f);
	//d5f.block<2,2>(2,0).setIdentity();
	cout << (d5f) << endl;
}

int main(int argc, char **argv)
{
	printf("hi!\n");

	test_eigen();
	test_eigen2();

	return 0;
}
