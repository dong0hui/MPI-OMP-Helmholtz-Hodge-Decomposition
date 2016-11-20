#include "gf2dhhd.h"

//2D Green Function
void gf2d::aperiodic(long double dx, long double dy, long double delta){
    int m_quarter=(m+1)/2;
    int n_quarter=(n+1)/2;
	long double pi = 1/3.141592653589793;
	int i=0,j=0;
	
	for (i=0; i<m_quarter; i++){
		for (j=0; j<n_quarter; j++){
		  if (i==0 && j==0){
		    gf[m_quarter-1][n_quarter-1] = -0.5*pi*log(sqrt(delta));
#if VERBOSE >=1
				printf("%f ",gf[m_quarter-1][n_quarter-1]);
#endif
		  }
		  else {  //dy is vertical grid and applied to rows, while dx is horizontal grid and applied on column
			  gf[m_quarter-1+i][n_quarter-1+j] = -0.5*pi*log(sqrt(pow(i*dy,2)+pow(j*dx,2)));
#if VERBOSE >=1
			  printf("%f ",gf[m_quarter-1+i][n_quarter-1+j]);
#endif
		  }
		} 
#if VERBOSE >=1
		cout<<endl;
#endif
	}

	for (i=0; i<m_quarter-1; i++){
		for (j=0; j<n_quarter; j++){
			gf[i][n_quarter-1+j]=gf[m-1-i][n_quarter-1+j];
		}
	}

	for (i=0; i<m; i++){
		for (j=0; j<n_quarter-1; j++){
			gf[i][j] = gf[i][n-1-j];
		}
	}
}
void gf2d::periodic(long double dx, long double dy, int num_period, long double delta, int ntds){
	if (num_period%2 == 0) {cout<<"Number of period has to be odd because period is truncated evenly...\n"; exit(1);}
	long double ax = dx*(0.5*(n-1)); //ax is lattice constant along x-direction, n is odd, gf2d is 2x2 of unit cell
	long double ay = dy*(0.5*(m-1));
	long double *rx_pb = new long double [num_period];
	for (int i=0; i<num_period; i++) rx_pb[i] = (i-(num_period-1)/2)*ax; //-2ax,-ax,0,ax,2ax if num_period=5
	long double *ry_pb = new long double [num_period];
	for (int i=0; i<num_period; i++) ry_pb[i] = (i-(num_period-1)/2)*ay;
	
	int m_quarter = (m+1)/2;
	int n_quarter = (n+1)/2;
	long double pi = 1/3.141592653589793;
	long double subGF;
	int i,j;
	omp_set_num_threads(ntds);
#pragma omp parallel num_threads(ntds) default(shared) private(subGF,j)
#pragma omp for schedule(static)
	for (i = 0; i<m_quarter; i++){
		for (j = 0; j<n_quarter; j++){
		    //gf[m_quarter-1+i][n_quarter-1+j] = 0.0;
			//subGF = 0.0;
			for (int i1 = 0; i1<num_period; i1++){
				for (int i2=0; i2<num_period; i2++){
					subGF = -0.5*pi*log(sqrt(pow(i*dy+ry_pb[i1],2)+pow(j*dx+rx_pb[i2],2)));
					if (isinf(subGF)) subGF = -0.5*pi*log(sqrt(delta));
					gf[m_quarter-1+i][n_quarter-1+j] += subGF;
				}
			}
#if VERBOSE >=1
			printf("%f ",gf[m_quarter-1+i][n_quarter-1+j]);
#endif		       
		}
#if VERBOSE >=1
		cout<<endl;
#endif
	}
	delete[] rx_pb;
	delete[] ry_pb;
	
#pragma omp parallel for num_threads(ntds) default(shared) private(j)
	for (i=0; i<m_quarter-1; i++){
		for (j=0; j<n_quarter; j++){
			gf[i][n_quarter-1+j]=gf[m-1-i][n_quarter-1+j];
		}
	}
#pragma omp parallel for num_threads(ntds) default(shared) private(j)
	for (i=0; i<m; i++){
		for (j=0; j<n_quarter-1; j++){
			gf[i][j] = gf[i][n-1-j];
		}
	}
	
}
void gf2d::output(char *str1){
	int i,j;
	ofstream fout(str1);
	if (!str1)
	{cout<<"Cannot create the file "<<str1<<endl; exit(1);}
#if VERBOSE >=1
	cout<<"Green's Function is:\n";
#endif
	for (i=0; i<m; i++){
		for (j=0; j<n; j++){
		fout<<" "<<std::setprecision(16)<<gf[i][j];
#if VERBOSE >=1
		cout<<" "<<gf[i][j];
#endif
		}
		fout<<endl; 
#if VERBOSE >=1
		cout<<endl;
#endif
	}
	fout.close();
}


//Find Gradient of 2D matrix
void mtxgdt::input(char *str1){
	int i,j;
	ifstream fin(str1);
	if (!fin)
	  {cout<<"Input identity "<<str1<<" doesn't exist--mtxgdt.c\n"; exit(1);}
	for (i=0; i<m; i++){
		for (j=0; j<n; j++)
			fin>>a[i][j];
	}
	fin.close();
#if VERBOSE >= 1
	for (i=0; i<m; i++){
	  for (j=0; j<n; j++)
	    cout<<a[i][j]<<" ";
	  cout<<endl;
	}
#endif
}
void mtxgdt::gdt(long double dx, long double dy, long double **a){
	int i,j;

	for (i=0; i<m; i++){
		for (j=1; j<n-1; j++){
			gdtx_a[i][j] = 0.0;
			gdtx_a[i][j] = 0.5*(a[i][j+1]-a[i][j-1])/dx;
		}
		gdtx_a[i][0] = 0.0; 
		gdtx_a[i][0]=(a[i][1]-a[i][0])/dx;
		gdtx_a[i][n-1] = 0.0;
		gdtx_a[i][n-1]=(a[i][n-1]-a[i][n-2])/dx;
	}
#if VERBOSE >= 1
	for (i=0; i<m; i++){
	  for (j=0; j<n; j++)
	    cout<<gdtx_a[i][j]<<" ";
	  cout<<endl;
	}
#endif

	for (j=0; j<n; j++){
		for (i=1; i<m-1; i++){
			gdty_a[i][j] = 0.0;
			gdty_a[i][j] = 0.5*(a[i+1][j]-a[i-1][j])/dy;
		}
		gdty_a[0][j] = 0.0;
		gdty_a[0][j] = (a[1][j]-a[0][j])/dy;
		gdty_a[m-1][j] = 0.0;
		gdty_a[m-1][j] = (a[m-1][j]-a[m-2][j])/dy;
	}
#if VERBOSE >= 1
	for (i=0; i<m; i++){
	  for(j=0; j<n; j++)
	    cout<<gdty_a[i][j]<<" ";
	  cout<<endl;
	}
#endif
}
void mtxgdt::output(char *str2,char *str3){
	int i,j;
	ofstream foutx(str2);
	if (!str2)
	{cout<<"Cannot open the file"<<str2<<endl;exit(1);}
#if VERBOSE >= 1
	cout<<"Gradient x-component is:\n";
#endif
	for (i=0; i<m; i++){
		for (j=0; j<n; j++){
			foutx<<" "<<std::setprecision(16)<<gdtx_a[i][j];
#if VERBOSE >= 1
			cout<<" "<<gdtx_a[i][j];
#endif
		}
		foutx<<endl; 
#if VERBOSE >= 1
		cout<<endl;
#endif
	}
	foutx.close();
	
	ofstream fouty(str3);
	if (!str3)
	{cout<<"Cannot open the file"<<str3<<endl;exit(1);}
#if VERBOSE >= 1
	cout<<"Gradient y-component is:\n";
#endif
	for (i=0; i<m; i++){
		for(j=0; j<n; j++){
			fouty<<" "<<std::setprecision(16)<<gdty_a[i][j];
#if VERBOSE >= 1
			cout<<" "<<gdty_a[i][j];
#endif
		}
		fouty<<endl; 
#if VERBOSE >= 1
		cout<<endl;
#endif
	}
	fouty.close();
}


//Find Sum of 2D Matrix, This class is not necessary
void mtxsum::input(char *str1){
        int i,j;
	ifstream fin(str1);
	if (!fin)
	{cout<<"Input matrix "<<str1<<" doesn't exist--mtxsum.c/n";exit(1);}
	for (i=0; i<m; i++){
		for (j=0; j<n; j++)
			fin>>a[i][j];
	}
	fin.close();
	
#if VERBOSE >= 1
	for (i=0; i<m; i++){
		for (j=0; j<n; j++)
			cout<<a[i][j]<<" ";
		cout<<endl;
	}
#endif
}
long double mtxsum::sumsum(){
	int i,j;
	sum = 0.0;
	for (i=0; i<m; i++){
		for (j=0; j<n; j++)
			sum += a[i][j];
	}
#if VERBOSE >= 1
	cout<<sum<<endl;
#endif
	return sum;
}


//Calculate the Potentials, MPI to split irow
void potentials::cal_potentials(int rowstart, int rowend, int mglobal, int nglobal, long double dx, long double dy, long double **Vx, long double **Vy, long double **dGx, long double **dGy, int ntds){
		
	omp_set_num_threads(ntds);	
#pragma omp parallel for num_threads(ntds)
	for(int irow=rowstart; irow<=rowend; irow++){
		for (int icolumn=0; icolumn<nglobal; icolumn++){
			long double sumphi = 0.0, sumpsi = 0.0;
			for (int i=0; i<mglobal; i++){
				for(int j=0; j<nglobal; j++){
					sumphi += Vx[i][j]*dGx[mglobal-irow+i][nglobal-icolumn+j]+Vy[i][j]*dGy[mglobal-irow+i][nglobal-icolumn+j];
					sumpsi += Vx[i][j]*dGy[mglobal-irow+i][nglobal-icolumn+j]-Vy[i][j]*dGx[mglobal-irow+i][nglobal-icolumn+j];
				}
			}
			phi[irow-rowstart][icolumn] = sumphi*dx*dy;
			psi[irow-rowstart][icolumn] = sumpsi*dx*dy;
		}
	}
}
