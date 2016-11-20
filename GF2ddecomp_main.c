#include "gf2dhhd.h"
//COMPILE: mpicxx gf2dhhd.c gf2dhhd_main.c -o gf2dhhd_main -std=c99 -openmp -lrt -wd981 
//strategy: first, send and receive contiguous 2D matrices (Done!). Then, Write to the same file MPI-IO (Working)

int main(int argc, char **argv){
	
	MPI_Status status;
	MPI_File file_phi; //MPI-IO to the same output file
	MPI_File file_psi;
	MPI_Datatype local_phi; //Create subarray
	MPI_Datatype local_psi;
	char procname[32]; //store processor's name
	int ierr; ierr = MPI_Init(&argc,&argv); 
	int mpirank; ierr |= MPI_Comm_rank(MPI_COMM_WORLD, &mpirank); 
	int mpisize; ierr |= MPI_Comm_size(MPI_COMM_WORLD, &mpisize); 
	int namelen; MPI_Get_processor_name(procname, &namelen);      
	
	char *str1 = "Vx.dat";
	char *str2 = "Vy.dat";
	const int m=99,n=70; //Change!
	const long double delta = 4.832930238571752e-09;
	int num_period = (argc ==3)? atoi(argv[2]):20;
	long double width = 1.0, height=1.4;
	long double dx = width/n, dy = height/m; //changed, Note Vx and Vy are deleted 1 row and 1 column
	
	//Divide rows onto different processes
	const int root = 0;
	int local_mlen = m/mpisize; //4 proc, 105/4 = 26
	int rowstart   = mpirank*local_mlen; //0-25;26-51,...
	int rowend     = (mpirank+1)*local_mlen-1;
	if (mpirank == mpisize-1){ //78-105
		rowend = m-1; local_mlen = rowend-rowstart+1;
	}
	
	//To use MPI_Bcast, these 4 arrays are contiguous
	long double **Vx, **Vy;
	long double **dGx, **dGy;
	Vx = new long double *[m]; Vy = new long double *[m];
	Vx[0] = new long double [m*n]; Vy[0] = new long double [m*n];
	for (int i=1; i<m; i++){
		Vx[i] = &Vx[0][i*n];
		Vy[i] = &Vy[0][i*n];
	}
	dGx = new long double *[2*m+1]; dGy = new long double *[2*m+1];
	dGx[0] = new long double [(2*m+1)*(2*n+1)];
	dGy[0] = new long double [(2*m+1)*(2*n+1)];
	for (int i=1; i<2*m+1; i++){
		dGx[i] = &dGx[0][i*(2*n+1)];
		dGy[i] = &dGy[0][i*(2*n+1)];
	}
	
	int ntds = (argc == 2)? atoi(argv[1]):6; //OpenMP threads number
	struct timespec tcgstart = {0,0}; //time of constructing GF
	struct timespec tcgend = {0,0};
	struct timespec tgdtstart = {0,0}; //time of finding gradient
	struct timespec tgdtend = {0,0};
	struct timespec tpstart = {0,0}; //time of finding potential
	struct timespec tpend = {0,0};
	
	//Vx and Vy can be read from the same file by different processors
	if (mpirank == 0) {
		
		//read force field into memory
		ifstream Vxin(str1);
		if (!Vxin) {cout<<"Input file Vx does not exist!\n"; exit(1);}
		for (int i=0; i<m; i++){
			for (int j=0; j<n; j++){
				Vxin>>Vx[i][j];
			}
		}
		Vxin.close();
		ifstream Vyin(str2);
		if(!Vyin) {cout<<"Input file Vy does not exist!\n"; exit(1);}
		for (int i=0; i<m; i++){
			for (int j=0; j<n; j++){
				Vyin>>Vy[i][j];
			}
		}
		Vyin.close();
		
		//Build 2x larger Green Function, use OpenMP
		gf2d GreenF(2*m+1,2*n+1);
		clock_gettime(CLOCK_REALTIME,&tcgstart);
		GreenF.periodic(dx,dy,2*num_period+1,delta,ntds); //if small, OpenMP, if large OpenMP+MPI (Will See)
		clock_gettime(CLOCK_REALTIME,&tcgend);
		cout<<"Time of building Green's Function is "<<(tcgend.tv_sec-tcgstart.tv_sec)*1000+(tcgend.tv_nsec-tcgstart.tv_nsec)/1000000<<"ms\n";
#if VERBOSE >=1		
		GreenF.output("GreenF100.dat");
#endif		
		//Find Gradient of Green's Function, not time-consuming, no need to parallelize
		mtxgdt GF(2*m+1,2*n+1);
#if VERBOSE >=1
		GF.input("GreenF100.dat");
#endif
		clock_gettime(CLOCK_REALTIME,&tgdtstart);
		GF.gdt(dx,dy,GreenF.gf); //OpenMP
		clock_gettime(CLOCK_REALTIME,&tgdtend);
		cout<<"Time of finding the gradient is "<<(tgdtend.tv_sec-tgdtstart.tv_sec)*1000+(tgdtend.tv_nsec-tgdtstart.tv_nsec)/1000000<<"ms\n";
#if VERBOSE >=1
		GF.output("dGx100.dat","dGy100.dat");
#endif		
		//Read dGx.dat and dGy.dat into dGx and dGy
		for (int i=0; i<2*m+1; i++) dGx[i] = &GF.gdtx_a[0][i*(2*n+1)];
		for (int i=0; i<2*m+1; i++) dGy[i] = &GF.gdty_a[0][i*(2*n+1)];
		cout<<"Master's Rank which is "<<mpirank<<"; and name is "<<procname<<" has finished calculating Vx, Vy, dGx, and dGy!"<<endl;
	}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	
		
	//Now we have Vx, Vy, dGx, dGy, they should be broadcasted to every process
	MPI_Bcast(&Vx[0][0], m*n, MPI_LONG_DOUBLE, root, MPI_COMM_WORLD);
	MPI_Bcast(&Vy[0][0], m*n, MPI_LONG_DOUBLE, root, MPI_COMM_WORLD);
	MPI_Bcast(&dGx[0][0], (2*m+1)*(2*n+1), MPI_LONG_DOUBLE, root, MPI_COMM_WORLD);
	MPI_Bcast(&dGy[0][0], (2*m+1)*(2*n+1), MPI_LONG_DOUBLE, root, MPI_COMM_WORLD);
	
	//This part is the most time consuming part, no need for barrier
	potentials ptl(local_mlen,n);
	clock_gettime(CLOCK_REALTIME,&tpstart);
	ptl.cal_potentials(rowstart, rowend, m, n, dx, dy, Vx, Vy, dGx, dGy, ntds);
	clock_gettime(CLOCK_REALTIME,&tpend);
	MPI_Barrier(MPI_COMM_WORLD); //not necessary, just want to see print screen
	fflush(stdout);
	cout<<"The rank "<<mpirank<<" whose name is "<<procname<<" calculates rows from "<<rowstart<<" to "<<rowend<<";"<<endl;
	cout<<"\tTime of finding the potential is "<<(tpend.tv_sec-tpstart.tv_sec)*1000+(tpend.tv_nsec-tpstart.tv_nsec)/1000000<<"ms\n";

	//Different processes write to the same "phi.txt" and "psi.txt" simultaneously-LAST PROBLEM
	//Create char datatype
	char phi_file[30], psi_file[30];
	sprintf(phi_file, "phi_p%d.dat", num_period);
	sprintf(psi_file, "psi_p%d.dat", num_period);
	MPI_Datatype phi_as_string;
	const int charspernum = sizeof(long double)+8;
	MPI_Type_contiguous(charspernum, MPI_CHAR, &phi_as_string); 
	MPI_Type_commit(&phi_as_string);
	char *phi_as_txt = new char [local_mlen*n*charspernum];
	int count = 0;
	for (int i=0; i<local_mlen; i++){
		for (int j=0; j<n-1; j++){
			sprintf(&phi_as_txt[count*charspernum], "%1.15Le ", ptl.phi[i][j]);
			count++;
		}
		sprintf(&phi_as_txt[count*charspernum], "%1.15Le\n", ptl.phi[i][n-1]);
		count++;
	}
	
	MPI_Datatype psi_as_string;
	MPI_Type_contiguous(charspernum, MPI_CHAR, &psi_as_string); 
	MPI_Type_commit(&psi_as_string);
	char *psi_as_txt = new char [local_mlen*n*charspernum];
	count = 0;
	for (int i=0; i<local_mlen; i++){
		for (int j=0; j<n-1; j++){
			sprintf(&psi_as_txt[count*charspernum], "%1.15Le ", ptl.psi[i][j]);
			count++;
		}
		sprintf(&psi_as_txt[count*charspernum], "%1.15Le\n", ptl.psi[i][n-1]);
		count++;
	}
	
	//Create subarray
	int globalsizes[2] = {m,n};
	int localsizes[2]  = {local_mlen,n};
	int starts[2]      = {rowstart, 0};
	int order		   = MPI_ORDER_C; 
	MPI_Type_create_subarray(2, globalsizes, localsizes, starts, order, phi_as_string, &local_phi);
	MPI_Type_commit(&local_phi);
	MPI_Type_create_subarray(2, globalsizes, localsizes, starts, order, psi_as_string, &local_psi);
	MPI_Type_commit(&local_psi);
	
	MPI_File_open(MPI_COMM_WORLD,phi_file,
				  MPI_MODE_CREATE|MPI_MODE_WRONLY, //WRONLY:Write only
				  MPI_INFO_NULL, &file_phi);
	MPI_File_set_view(file_phi, 0, MPI_CHAR, local_phi, "native", MPI_INFO_NULL);
	MPI_File_write_all(file_phi, phi_as_txt, local_mlen*n, phi_as_string, &status);
	MPI_File_close(&file_phi);
	
	MPI_File_open(MPI_COMM_WORLD,psi_file,
				  MPI_MODE_CREATE|MPI_MODE_WRONLY, //WRONLY:Write only
				  MPI_INFO_NULL, &file_psi);
	MPI_File_set_view(file_psi, 0, MPI_CHAR, local_psi, "native", MPI_INFO_NULL);
	MPI_File_write_all(file_psi, psi_as_txt, local_mlen*n, psi_as_string, &status);
	MPI_File_close(&file_psi);
	
	MPI_Type_free(&local_phi);
	MPI_Type_free(&local_psi);
	MPI_Type_free(&phi_as_string);
	MPI_Type_free(&psi_as_string);
	
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	
	delete[] Vx;
	delete[] Vy;
	delete[] dGx;
	delete[] dGy;
	delete[] phi_as_txt;
	delete[] psi_as_txt;

	MPI_Finalize();
	return 0;
}
