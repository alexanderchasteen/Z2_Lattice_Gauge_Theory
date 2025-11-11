#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

/* the lattice is of dimensions SIZE**4 */
#define SIZE 5
#define CONFIG 20    /* Number of parallel tempering configurations */
int link[SIZE][SIZE][SIZE][SIZE][4][CONFIG]; /* second last index gives link direction, last index indexes the configuration */ 


/* utility functions */
void moveup(int x[],int d) {
    x[d]+=1;
    if (x[d]>=SIZE) x[d]-=SIZE;
    return;
}
void movedown(int x[],int d) {
    x[d]-=1;
    if (x[d]<0) x[d]+=SIZE;
    return;
}
void coldstart(){/* set all links to unity in every lattice */
    int x[4],d,lattice_config;
    for (lattice_config=0; lattice_config<CONFIG; lattice_config++){
        for (x[0]=0;x[0]<SIZE;x[0]++)
            for (x[1]=0;x[1]<SIZE;x[1]++)
                for (x[2]=0;x[2]<SIZE;x[2]++)
                    for (x[3]=0;x[3]<SIZE;x[3]++)
                        for (d=0;d<4;d++)
                            link[x[0]][x[1]][x[2]][x[3]][d][lattice_config]=1;
    

    }
   return;
}

void swap_double(double* array, int index1, int index2){
    double tmp=array[index1];
    array[index1]=array[index2];
    array[index2]=tmp;
}

void swap_int(int* array, int index1, int index2){
    int tmp=array[index1];
    array[index1]=array[index2];
    array[index2]=tmp;
}

/* for a random start: call coldstart() and then update once at beta=0 */
/* do a Monte Carlo sweep; return action */
/* sweeping throgh all nodes in the lattice (first 4 nums) and the 3rd number d is the direction on that node to give the edge*/
/* Then move across the perpendicular directions and collect the action and determine if based on the boltzman probability you update the link*/
double update(double beta,int lattice_config){ 
    int x[4],d,dperp,staple,staplesum;
    double bplus,bminus,action=0.0;
    for (x[0]=0; x[0]<SIZE; x[0]++)
        for (x[1]=0; x[1]<SIZE; x[1]++)
            for (x[2]=0; x[2]<SIZE; x[2]++)
                for (x[3]=0; x[3]<SIZE; x[3]++)
                    for (d=0; d<4; d++) {
                        staplesum=0;
                        for (dperp=0;dperp<4;dperp++){
                            if (dperp!=d){
                                int y[4]; for(int i=0;i<4;i++) y[i]=x[i];
                                movedown(y,dperp);
                                staple=link[y[0]][y[1]][y[2]]
                                [y[3]][dperp][lattice_config]*link[y[0]][y[1]][y[2]][y[3]][d][lattice_config];
                                moveup(y,d);
                                staple*=link[y[0]][y[1]][y[2]][y[3]][dperp][lattice_config];
                                moveup(y,dperp);
                                staplesum+=staple;
                                /* plaquette 1456 */
                                staple=link[y[0]][y[1]][y[2]][y[3]][dperp][lattice_config];
                                moveup(y,dperp);
                                movedown(y,d);
                                staple*=link[y[0]][y[1]][y[2]][y[3]][d][lattice_config];
                                movedown(y,dperp);
                                staple*=link[y[0]][y[1]][y[2]][y[3]][dperp][lattice_config];
                                staplesum+=staple;
                            }
                        }
                        bplus=exp(beta*staplesum);
                        bminus=1/bplus;
                        bplus=bplus/(bplus+bminus);

            /* the heatbath algorithm */
                        if (drand48() < bplus){
                            link[x[0]][x[1]][x[2]][x[3]][d][lattice_config]=1;
                            action+=staplesum;
                        }
                        else {
                            link[x[0]][x[1]][x[2]][x[3]][d][lattice_config]=-1;
                            action-=staplesum;
                        }
                    }
    action/=(SIZE*SIZE*SIZE*SIZE*4*6);
    return action;
}

void thermalize(int thermal_sweeps, double beta,int lattice_config){
    for (int i=0; i<thermal_sweeps; i++){
        update(beta,lattice_config);
    }
}


double mean(double *data, int N) {
    double s = 0.0;
    for (int i = 0; i < N; i++){
        s += data[i];
    }    
    return s / N;
}

double variance(double *data, int N) {
    double m=mean(data,N);
    double s = 0.0;
    for (int i = 0; i < N; i++) {
        double d = data[i] - m;
        s += d * d;
    }
    return s / N;
}
void autocorr(double *data, int N, int maxlag, double *rho) {
    double m=mean(data,N);
    double var = variance(data, N);
    if (var == 0.0) {
        for (int t=0; t<maxlag; t++) rho[t] = (t==0)?1.0:0.0;
        return;
    }

    for (int t = 0; t < maxlag; t++) {
        double c = 0.0;
        for (int i = 0; i < N - t; i++) {
            c += (data[i] - m) * (data[i + t] - m);
        }
        c /= (N - t);   
        rho[t] = c / var;
    }
}

double tau_int(double *rho, int maxlag) {
    double tau = 1;
    for (int t = 1; t < maxlag; t++) {
     tau += 2*  rho[t];
    }
    return tau;
}

int main(){
    double beta, dbeta, action,volume,beta_min,beta_max,svendsen_length;
    int thermal_sweeps, measurment_sweeps, maxlag, autocorrelation_sweeps,cols,blocks;
    
    srand48(1234L);
    srand(1234);

    dbeta=.00025;
    thermal_sweeps=1000000;  
    autocorrelation_sweeps=10000;  
    measurment_sweeps=10000;  
    maxlag=8000;
    volume=SIZE*SIZE*SIZE*SIZE*6; 
    beta_min=0.436;
    beta_max=beta_min+CONFIG*dbeta;
    blocks=50;



    FILE *fp4;
    char filename4[100];
    // 
    sprintf(filename4, "PT%d_RawMCfine.000251mil.csv", SIZE);
    fp4=fopen(filename4,"w");

    // Finer corresponds to dbeta=0.001 instead of 0.005 like b4
             
    for (double x=beta_min;x<beta_max-dbeta; x+=dbeta){
        fprintf(fp4,"%g,",x); 
    }
    fprintf(fp4,"%g \n",beta_max-dbeta);


    double beta_array[CONFIG];

    for (int i=0;i<CONFIG;i++){
        beta_array[i]=beta_min+i*dbeta;
    }
    



    int beta_index_array[CONFIG];
    
    for (int i=0;i<CONFIG;i++){
        beta_index_array[i]=i;
    }
    
    
    double action_array[CONFIG];
    double autocorrelation_array[autocorrelation_sweeps];

    double IAT_array[CONFIG];
    double rho[maxlag];

    coldstart();

    //Thermalize the configurations with parallel tempering
    int counter=0;
    for (int j=0;j<thermal_sweeps;j++){
        counter+=1;
        if (counter % 10000==0){
            printf("Thermalization sweep %d out of %d \n",counter,thermal_sweeps);
        }

        for (int i=0; i<CONFIG;i++){
            double beta=beta_array[i];
            action=update(beta,i);
            action_array[i]=action;
           
        }


        // for (int i = 0; i < CONFIG - 2; i++) {
        //     double delta = volume * (beta_array[i] - beta_array[i + 1]) *(action_array[i + 1] - action_array[i]);
        //     double Pswap = exp(delta);
        //     if (Pswap > 1.0) Pswap = 1.0;
        //     double x = drand48();
        //     if (x < Pswap) {
        //         swap_double(beta_array, i, i + 1);
        //         swap_int(beta_index_array, i, i + 1);
        //         }
        //     }

        // double delta = volume * (beta_array[CONFIG-1] - beta_array[0]) *(action_array[0] - action_array[CONFIG-1]);
        // double Pswap = exp(delta);
        // if (Pswap > 1.0) Pswap = 1.0;
        // double x = drand48();
        // if (x < Pswap) {
        //     swap_double(beta_array, CONFIG-1, 0);
        //     swap_int(beta_index_array, CONFIG-1, 0);
        //     }




        for (int i=0; i<CONFIG;i++){
            if (i != CONFIG-1){
                for (int j=i+1; j<CONFIG; j++){
                    double delta=volume*(beta_array[i]-beta_array[j])*(action_array[j]-action_array[i]);
                    double Pswap=exp(delta);
                    if (Pswap>1.0) Pswap=1.0;
                    double x=drand48();
                    if (x<Pswap){
                        swap_double(beta_array,i,j);
                        swap_int(beta_index_array,i,j);
                        swap_double(IAT_array,i,j);
                        }
                }
            }
        }

        
    }

    // Autocorelation sweeps

    for (int i=0; i<CONFIG;i++){
        for (int j=0; j<autocorrelation_sweeps;j++){
            double beta=beta_array[i];
            action=update(beta,i);
            autocorrelation_array[j]=action;
        }
        autocorr(autocorrelation_array, autocorrelation_sweeps, maxlag,rho);
        double taucomp=tau_int(rho,maxlag);
        if (taucomp < 1.0) taucomp = 1.0; /*safety*/ 
        double tau=ceil(taucomp);
        IAT_array[i]=tau;
        printf("Beta %g  IAT %g \n",beta_array[i],tau);
    }



    //Measurement sweeps with parallel tempering
    for (int j=0; j<measurment_sweeps;j++){
        for (int i=0; i<CONFIG;i++){
            double beta=beta_array[i];
            thermalize(IAT_array[i]-1,beta,i);
            action=update(beta,i);
            action_array[i]=action;   
        }

        // for (int i = 0; i < CONFIG - 2; i++) {
        //     double delta = volume * (beta_array[i] - beta_array[i + 1]) *(action_array[i + 1] - action_array[i]);
        //     double Pswap = exp(delta);
        //     if (Pswap > 1.0) Pswap = 1.0;
        //     double x = drand48();
        //     if (x < Pswap) {
        //         swap_double(beta_array, i, i + 1);
        //         swap_int(beta_index_array, i, i + 1);
        //         }
        //     }

        // double delta = volume * (beta_array[CONFIG-1] - beta_array[0]) *(action_array[0] - action_array[CONFIG-1]);
        // double Pswap = exp(delta);
        // if (Pswap > 1.0) Pswap = 1.0;
        // double x = drand48();
        // if (x < Pswap) {
        //     swap_double(beta_array, CONFIG-1, 0);
        //     swap_int(beta_index_array, CONFIG-1, 0);
        //     swap_double(IAT_array, CONFIG-1, 0);
        //     }

        for (int i=0; i<CONFIG;i++){
            if (i != CONFIG-1){
                for (int j=i+1; j<CONFIG; j++){
                    double delta=volume*(beta_array[i]-beta_array[j])*(action_array[j]-action_array[i]);
                    double Pswap=exp(delta);
                    if (Pswap>1.0) Pswap=1.0;
                    double x=drand48();
                    if (x<Pswap){
                        swap_double(beta_array,i,j);
                        swap_int(beta_index_array,i,j);
                        swap_double(IAT_array,i,j);
                        }
                }
            }
        }  
        for (int i=0; i<CONFIG; i++){
            int beta_i_index;
            for (int k=0; k<CONFIG;k++){
                if (i==beta_index_array[k]){    
                    beta_i_index=k;
                    // break;
                }   
            }
            

            double x=action_array[beta_i_index];
            fprintf(fp4, "%g",x);
            if (i<CONFIG-1) fprintf(fp4, ",");


        
        
        }
        fprintf(fp4, "\n");

        }





    fclose(fp4);
    
    
    

    /* =========================== */
   
    exit(0);
}
