#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/* the lattice is of dimensions SIZE**4 */
#define SIZE 5
int link[SIZE][SIZE][SIZE][SIZE][4]; /* last index gives link direction */


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
void coldstart(){/* set all links to unity */
    int x[4],d,q;
    for (x[0]=0;x[0]<SIZE;x[0]++)
        for (x[1]=0;x[1]<SIZE;x[1]++)
            for (x[2]=0;x[2]<SIZE;x[2]++)
                for (x[3]=0;x[3]<SIZE;x[3]++)
                    for (d=0;d<4;d++)
                        link[x[0]][x[1]][x[2]][x[3]][d]=1;
    return;
}


/* for a random start: call coldstart() and then update once at beta=0 */
/* do a Monte Carlo sweep; return energy */
/* sweeping throgh all nodes in the lattice (first 4 nums) and the 3rd number d is the direction on that node to give the edge*/
/* Then move across the perpendicular directions and collect the action and determine if based on the boltzman probability you update the link*/
double update(double beta){ 
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
                                [y[3]][dperp]*link[y[0]][y[1]][y[2]][y[3]][d];
                                moveup(y,d);
                                staple*=link[y[0]][y[1]][y[2]][y[3]][dperp];
                                moveup(y,dperp);
                                staplesum+=staple;
                                /* plaquette 1456 */
                                staple=link[y[0]][y[1]][y[2]][y[3]][dperp];
                                moveup(y,dperp);
                                movedown(y,d);
                                staple*=link[y[0]][y[1]][y[2]][y[3]][d];
                                movedown(y,dperp);
                                staple*=link[y[0]][y[1]][y[2]][y[3]][dperp];
                                staplesum+=staple;
                            }
                        }
                        bplus=exp(beta*staplesum);
                        bminus=1/bplus;
                        bplus=bplus/(bplus+bminus);

            /* the heatbath algorithm */
                        if (drand48() < bplus){
                            link[x[0]][x[1]][x[2]][x[3]][d]=1;
                            action+=staplesum;
                        }
                        else {
                            link[x[0]][x[1]][x[2]][x[3]][d]=-1;
                            action-=staplesum;
                        }
                    }
    action/=(SIZE*SIZE*SIZE*SIZE*4*6);
    return action;
}

void thermalize(int thermal_sweeps, double beta){
    for (int i=0; i<thermal_sweeps; i++){
        update(beta);
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

double jacknife_variance(double *data, int N) {
    double *jacknife_samples = malloc(N * sizeof(double));
    double var=0.0;
    for (int i=0; i<N; i++){
        double v=0;
        for (int j=0; j<N; j++){
            if (i==j) v+=0;
            else v+=data[j];
        }
        v=v/(double)(N-1);
        jacknife_samples[i]=v;          
    }
    double jacknifvar=(double)(N-1)*variance(jacknife_samples,N);
    free(jacknife_samples); // free inside function
    return jacknifvar;    
}


double* jacknifesamples(double *data, int N) {
    double *jacknife_samples = malloc(N * sizeof(double));
    for (int i=0; i<N; i++){
        double v=0;
        for (int j=0; j<N; j++){
            if (i==j) v+=0;
            else v+=data[j];
        }
        v=v/(double)(N-1);
        jacknife_samples[i]=v;          
    }
    return jacknife_samples;    
}



int main(){
    double beta, dbeta, action,volume,beta_min,beta_max,svendsen_length;
    int thermal_sweeps, measurment_sweeps, maxlag, autocorrelation_sweeps,cols,blocks;
    double *data, *rho, *independent_data;
    srand48(1234L);

    dbeta=.0005;
    thermal_sweeps=1000005;  
    autocorrelation_sweeps=100000;  
    measurment_sweeps=100000;  
    maxlag=80000;
    volume=SIZE*SIZE*SIZE*SIZE*6; 
    beta_min=0.43913;
    beta_max=0.43919;
    cols=(int)round((beta_max-beta_min)/dbeta)+1;
    blocks=50;
    svendsen_length=(int)round(dbeta/2*100000);
    
    /* ====== MEMORY ALLOCATION ====== */
    data = malloc(autocorrelation_sweeps * sizeof(double));
    rho = malloc(maxlag * sizeof(double));
    independent_data = malloc(measurment_sweeps * sizeof(double));

    double *intermediate_data_storage = malloc(measurment_sweeps * sizeof(double));
    double *avg_plaq_blocking = malloc(blocks * sizeof(double));
    double *sus_blocking = malloc(blocks * sizeof(double));
    double *binder_cum_blocking=malloc(blocks * sizeof(double));
    double *weights = malloc(measurment_sweeps * sizeof(double));
    double *logw = malloc(measurment_sweeps * sizeof(double));
    double *avg_plaq_svendsen_array = malloc(measurment_sweeps * sizeof(double));
    double *avg_plaq_sus_svendsen_array = malloc(measurment_sweeps * sizeof(double));
    double *plaq_blocking_svendsen = malloc(blocks * sizeof(double));
    double *sus_blocking_svendsen = malloc(blocks * sizeof(double));
    double* sus_blocking_binder=malloc(blocks*sizeof(double));
    double **dynamicMatrix = (double **)malloc(measurment_sweeps * sizeof(double *));
    for (int i = 0; i < measurment_sweeps; i++) {
    dynamicMatrix[i] = (double *)malloc(cols * sizeof(double));
}
    /* ================================ */

    FILE *fp;
    char filename[100];
    sprintf(filename, "Lattice_Size%d_TS%d_MS%d,AS%d.csv", SIZE, thermal_sweeps, measurment_sweeps,autocorrelation_sweeps);
    fp=fopen(filename, "w");
    fprintf(fp, "Beta,Avg Plaq,Plaq Sus,Jacknif Action Var,Jacknife Sus Var ,binder cum,Jacknife Binder cum,tau\n");

   FILE *fp2;
    char filename2[100];

    sprintf(filename2, "Swendsen and Ferrenberg Sweeps_Lattice_Size%d_TS%d_MS%d,AS%d.csv", SIZE, thermal_sweeps, measurment_sweeps,autocorrelation_sweeps);
    fp2 = fopen(filename2, "w");
    fprintf(fp2, "beta,avg_plaq,plaq_var,avg_sus,sus_var,binder_cum,binder_var\n");

    FILE *fp3;
    char filename3[100];

    sprintf(filename3, "Susceptibility_Jacknife_Samples_Lattice_Size%d_TS%d_MS%d,AS%d.csv", SIZE, thermal_sweeps, measurment_sweeps,autocorrelation_sweeps);
    fp3 = fopen(filename3, "w");


    FILE *fp4;
    char filename4[100];
    sprintf(filename4, "Raw_Monte_Carlo_Measurment%d_TS%d_MS%d,AS%d.csv", SIZE, thermal_sweeps, measurment_sweeps,autocorrelation_sweeps);
    fp4=fopen(filename4,"w");
             
    for (double x=beta_min;x<beta_max-dbeta; x+=dbeta){
        fprintf(fp4,"%g,",x); 
    }
    fprintf(fp4,"%g \n",beta_max-dbeta);

    /* heat it up */
    // Looking at a single beta 
    
    coldstart();
    thermalize(10000,beta_min);
    for (beta=beta_min; beta<beta_max; beta+=dbeta){
        thermalize(thermal_sweeps,beta);
        action=update(beta);
        /*take an autocorrelation sweep to compute monte carlo time tau*/
        for (int i=0; i<autocorrelation_sweeps; i++){
            action=update(beta);
            data[i]=action;
        }
        autocorr(data, autocorrelation_sweeps,maxlag,rho);
        double taucomp=tau_int(rho,maxlag);
        if (taucomp < 1.0) taucomp = 1.0; /*safety*/ 
        double tau=ceil(taucomp); 
        

        /*implement montecarlo time to take proper sample. This will make an arrat of size measurment_sweeps ot data*/
        for (int j=0; j<measurment_sweeps; j+=1){
            thermalize(tau,beta); 
            action=update(beta);
            intermediate_data_storage[j]=action; 
            int target_col=(int)round((beta-beta_min)/dbeta);
            dynamicMatrix[j][target_col]=action;
        }
     
        int sizeofblock=measurment_sweeps/blocks;
        for (int q=0; q<blocks; q++){
            double blockqdata=0;
            for (int p=0; p<sizeofblock; p++){
                blockqdata+=intermediate_data_storage[sizeofblock*q+p];
            }
            avg_plaq_blocking[q]=blockqdata/sizeofblock;
        }

        double avg=mean(avg_plaq_blocking,blocks);
        double jacknife_avg_plaq=jacknife_variance(avg_plaq_blocking,blocks);
        

        for (int q=0; q<blocks; q++){
            double blockqdata=0;
            for (int p=0; p<sizeofblock; p++){
                double diff=(intermediate_data_storage[sizeofblock*q+p]-avg);
                blockqdata+=diff*diff;
            }
            sus_blocking[q]=blockqdata/sizeofblock;
        }
       
        double plaq_sus=mean(sus_blocking,blocks);
        double jacknife_plaq_sus=jacknife_variance(sus_blocking,blocks);

         for (int q=0; q<blocks; q++){
            double second_mode=0;
            double fourth_mode=0;
            for (int p=0; p<sizeofblock; p++){
                double diff=(intermediate_data_storage[sizeofblock*q+p]-avg);
                second_mode+=diff*diff;
                fourth_mode+=diff*diff*diff*diff;
            }
            second_mode=second_mode/sizeofblock;
            fourth_mode=fourth_mode/sizeofblock;
            double blockqdata=1-fourth_mode/(3*second_mode*second_mode);
            binder_cum_blocking[q]=blockqdata;
        }
        double binder_cum=mean(binder_cum_blocking,blocks);
        double binder_cum_sus=jacknife_variance(binder_cum_blocking,blocks);
        
        printf("%g \t \t %g \t \t %g \t \t%g \t \t %g \t \t %g \t \t %g\n ",avg,plaq_sus,jacknife_avg_plaq,jacknife_plaq_sus,binder_cum,binder_cum_sus,tau);
        fprintf(fp, "%g,%g,%g,%g,%g,%g,%g,%g\n", beta,avg, plaq_sus, jacknife_avg_plaq,jacknife_plaq_sus,binder_cum,binder_cum_sus,tau);
        
        

        /*Now we will try to extend this using the ferrenberg svendsen method*/

        for (int q=-svendsen_length; q<svendsen_length;q++){
            //β'=β+i/100000 so we will start from β'=β-0.01 and end at β'=β+0.01
            // Start by computing the weights;
            double delta_beta=(double)q*.00001;
            double normalize=0.0;
            double maxlogw = -1e300;

            
            /* Step 1: Compute log weights to avoid overflow */
            for (int i = 0; i < measurment_sweeps; i++) {
                logw[i] = delta_beta * volume*intermediate_data_storage[i];   // sign is CRITICAL
                if (logw[i] > maxlogw) maxlogw = logw[i];
            }

    /* Step 2: Subtract max log weight to stabilize exponentials */
            double wsum = 0.0;
            for (int i = 0; i < measurment_sweeps; i++) {
                weights[i] = exp(logw[i] - maxlogw);
                wsum += weights[i];
            }

    
              
            double avg_plaq_svendsen=0.0;
            
            for (int j=0;j<measurment_sweeps; j++){
                avg_plaq_svendsen_array[j]=weights[j]*intermediate_data_storage[j];
                avg_plaq_svendsen+=weights[j]*intermediate_data_storage[j];
            }
            avg_plaq_svendsen/= wsum;
            // Fix weighting like this 
            
          
            for (int q=0; q<blocks; q++){
                double blockqdata=0;
                double weight=0.0;
                for (int p=0; p<sizeofblock; p++){
                    blockqdata+=avg_plaq_svendsen_array[sizeofblock*q+p];
                    weight+=weights[sizeofblock*q+p];
                }
                plaq_blocking_svendsen[q]=blockqdata/weight;
                            }
            
            double blocked_plaq_svendsen_variance=jacknife_variance(plaq_blocking_svendsen,blocks);
            

            

            double avg_plaq_sus_svendsen=0.0;
            
            for (int j=0;j<measurment_sweeps; j++){
                avg_plaq_sus_svendsen_array[j]=weights[j]*(intermediate_data_storage[j]-avg_plaq_svendsen)*(intermediate_data_storage[j]-avg_plaq_svendsen);
                avg_plaq_sus_svendsen+=weights[j]*(intermediate_data_storage[j]-avg_plaq_svendsen)*(intermediate_data_storage[j]-avg_plaq_svendsen);
            }
            
            avg_plaq_sus_svendsen /= wsum;

            for (int q=0; q<blocks; q++){
                double blockqdata=0;
                double weight=0.0;
                for (int p=0; p<sizeofblock; p++){
                    blockqdata+=avg_plaq_sus_svendsen_array[sizeofblock*q+p];
                    weight+=weights[sizeofblock*q+p];
                }
                sus_blocking_svendsen[q]=blockqdata/weight;
            }
          
            double blocked_sus_svendsen_variance=jacknife_variance(sus_blocking_svendsen,blocks);

            double binder_cum_svendsen=0;
            double* second_moment_array=malloc(measurment_sweeps *sizeof(double));
            double* fourth_moment_array=malloc(measurment_sweeps *sizeof(double));
            for (int j=0;j<measurment_sweeps; j++){
                double diff=intermediate_data_storage[j]-avg_plaq_svendsen;
                second_moment_array[j]=weights[j]*diff*diff;
                fourth_moment_array[j]=weights[j]*diff*diff*diff*diff;

            }
            double second_total = 0.0, fourth_total = 0.0;
            for (int j = 0; j < measurment_sweeps; j++) {
                second_total += second_moment_array[j];
                fourth_total += fourth_moment_array[j];
                }
            second_total /= wsum;
            fourth_total /= wsum;

            binder_cum_svendsen=1.0-(fourth_total)/(3*second_total*second_total);

            for (int q = 0; q < blocks; q++) {
                double block_m2 = 0.0;
                double block_m4 = 0.0;
                double wblock = 0.0;
                for (int p = 0; p < sizeofblock; p++) {
                    block_m2 += second_moment_array[q * sizeofblock + p];
                    block_m4 += fourth_moment_array[q * sizeofblock + p];
                    wblock += weights[q * sizeofblock + p];
                    }
                block_m2 /= wblock;
                block_m4 /= wblock;
                sus_blocking_binder[q] = 1.0 - block_m4 / (3.0 * block_m2 * block_m2);
            }
            
            double blocked_binder_svendsen_variance=jacknife_variance(sus_blocking_binder,blocks);

            // Take Jacknife Samples for Susceptibility at this Beta and add them to file3. 

            double* jack_sus_samples=jacknifesamples(sus_blocking_svendsen,blocks);
            fprintf(fp3, "%g,",beta+delta_beta);
            for (int i = 0; i < blocks; i++) {
                fprintf(fp3, "%.4f", jack_sus_samples[i]);  // print element
                if (i < 49) fprintf(fp3, ","); // add space except after last element
                }
            fprintf(fp3, "\n");  // optional: end line
            free(jack_sus_samples);


                
            fprintf(fp2, "%g,%g,%g,%g,%g,%g,%g\n", beta+delta_beta,avg_plaq_svendsen,blocked_plaq_svendsen_variance,avg_plaq_sus_svendsen,blocked_sus_svendsen_variance,binder_cum_svendsen,blocked_binder_svendsen_variance);
           
        }
       
        
        // Now we do somre more stuff with the intermediate storage particularly //       
    }
    for (int j = 0; j < measurment_sweeps; j++) {
    for (int i = 0; i < cols; i++) {
        fprintf(fp4, "%g", dynamicMatrix[j][i]);
        if (i < cols - 1) fprintf(fp4, ",");
    }
    fprintf(fp4, "\n");
}
  
   fclose(fp);
    fclose(fp2);
    fclose(fp3);
    fclose(fp4);
    
    /* ====== FREE MEMORY ====== */
    free(data);
    free(rho);
    free(independent_data);
    free(intermediate_data_storage);
    free(avg_plaq_blocking);
    free(sus_blocking);
    free(weights);
    free(logw);
    free(avg_plaq_svendsen_array);
    free(avg_plaq_sus_svendsen_array);
    free(plaq_blocking_svendsen);
    free(sus_blocking_svendsen);
    for (int i = 0; i < measurment_sweeps; i++) {
    free(dynamicMatrix[i]);
    }
    free(dynamicMatrix);
    free(sus_blocking_binder);

    /* =========================== */

    exit(0);
}
