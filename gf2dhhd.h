#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <omp.h>
#include <time.h>
#include <mpi.h>

using namespace std;
#define VERBOSE 0

class gf2d{
	private:
	int m,n;
	
	public:
	long double **gf;
	int ntds;
	gf2d(int mm, int nn){
		//to use MPI, 2D array has to be contiguous, gf[0] is initialized
		//gf[i] is set
		int i;
		m = mm; n = nn;
		if (m%2 == 0 || n%2 == 0)
		{cout<<"Green's Function must have odd rows and odd columns...\n";exit(1);}
		gf = new long double *[m];
		gf[0] = new long double[mm*nn];
		for (i=1; i<m; i++) gf[i]=&gf[0][i*n];
	}
	
	void aperiodic(long double,long double, long double);
	void periodic(long double, long double, int, long double, int);
	void output(char *);
	
	~gf2d(){
		delete[] gf;
	}
};

class mtxgdt{
	private:
	int m,n;
	
	public:
	long double **a, **gdtx_a, **gdty_a;
	mtxgdt(int mm, int nn){
		int i;
		m = mm; n = nn;
		a = new long double *[m];
		a[0] = new long double[mm*nn];
		for (i=1; i<m; i++) a[i] = &a[0][i*n];
		gdtx_a = new long double *[m];
		gdtx_a[0] = new long double[mm*nn];
		for (i=1; i<m; i++) gdtx_a[i] = &gdtx_a[0][i*n];
		gdty_a = new long double *[m];
		gdty_a[0] = new long double [mm*nn];
		for (i=1; i<m; i++) gdty_a[i] = &gdty_a[0][i*n];
	}
	void input(char *);
	void gdt(long double,long double,long double **);
	void output(char *, char *);
	~mtxgdt(){
		delete[] a;
		delete[] gdtx_a;
		delete[] gdty_a;
	}
};

class mtxsum{
	private:
	int m,n;
	
	public:
	long double **a, sum;
	mtxsum(int mm, int nn){
		int i;
		m=mm; n=nn;
		a = new long double *[m];
		for (i=0; i<m; i++) a[i] = new long double [n];
	}
	void input(char *);
	long double sumsum();
	~mtxsum(){
		int i;
		for (i=0; i<m; i++) {delete[] a[i];}
		delete[] a;
	}
};

class potentials{
	private:
	int m,n;
	
	public:
	long double **phi, **psi; //potentials.phi and potentials.psi are contiguous
	potentials(int mm, int nn){
		int i;
		m = mm; n=nn;
		phi = new long double *[m];
		phi[0] = new long double [m*n];
		for (i=1; i<m; i++) phi[i] = &phi[0][i*n];
		psi = new long double *[m];
		psi[0] = new long double [m*n];
		for (i=1; i<m; i++) psi[i] = &psi[0][i*n];
	}
	void cal_potentials(int, int, int, int, long double, long double, long double **, long double **, long double **, long double **, int);
	~potentials(){
		delete[] phi;
		delete[] psi;
	}
};
