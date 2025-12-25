#include <iostream>
#include "Headers.h"
#include <fstream>
#include <iomanip>

int main() {
    // File for the Raw Monte Carlo Data
    std::string filename1 = "Raw_MC_Data_M" + std::to_string(SIZE) + ".csv";
    std::ofstream MCfile(filename1);

    if (!MCfile) {
        std::cerr << "Error creating file!\n";
        return 1;
    }

    MCfile << std::fixed << std::setprecision(6);

    for (int i=0; i<CONFIG; i++){
        double x =beta_min+i * dbeta;
        MCfile << x;
        if (i < CONFIG-1) MCfile<<",";
    }
    MCfile<<"\n";
    std::cout << "File created.\n";
    // Done Making the Output File



    // File for Thermalization Information
    std::string filename2 = "Therm_Data_M" + std::to_string(SIZE) + ".csv";
    std::ofstream Thermfile(filename2);

    if (!Thermfile) {
        std::cerr << "Error creating file!\n";
        return 1;
    }

    Thermfile << std::fixed << std::setprecision(6);

    for (int i=0; i<CONFIG; i++){
        double x =beta_min+i * dbeta;
        Thermfile << x;
        if (i < CONFIG-1) Thermfile<<",";
    }
    Thermfile<<"\n";
    std::cout << "File created.\n";
    // Done Making the Output File


    



    // Make the Necessary Arrays to keep track of the things
    std::array<double,CONFIG> beta_array;
    for (int i=0; i <CONFIG; i++){
        beta_array[i]=beta_min+i*dbeta; 
    }
    std::array<int,CONFIG> beta_index_array;
    for (int i=0; i < CONFIG; i++){
        beta_index_array[i]=i;
    }


    std::array<double, CONFIG> action_array;
    std::array<double, autocorrelation_sweeps> autocorrelation_array;
    std::array<double, CONFIG> IAT_array;
    std::array<double, maxlag> rho;

    // Coldstart Initiation

    std::array<int, array_size> links;
    cold_start_array(links);
    


    // Thermalization Procedure 
   

    for (int j=0;j<thermal_sweeps; j++){
        
        for (int i=0; i<CONFIG; i++){
            double beta=beta_array[i];
            update_heatbath(links, beta,i);
            double action=update_metropolis(links, beta,i);
            action_array[i]=action;
        }

             
        for (int i=0; i<CONFIG-1;i++){
            double delta = volume * (beta_array[i]-beta_array[i+1])*(action_array[i+1]-action_array[i]);
            double Pswap=exp(delta);
            if (Pswap>1.0) Pswap=1.0;
            double x = sample_uniform();
            if (x < Pswap) {
                swap_double(beta_array, i, i + 1);
                swap_int(beta_index_array, i, i + 1);
                }
        }


        
        // for (int i=0; i<CONFIG;i++){
        //     if (i != CONFIG-1){
        //         for (int j=i+1; j<CONFIG; j++){
        //             double delta=volume*(beta_array[i]-beta_array[j])*(action_array[j]-action_array[i]);
        //             double Pswap=exp(delta);
        //             if (Pswap>1.0) Pswap=1.0;
        //             double x=sample_uniform();
        //             if (x<Pswap){
        //                 swap_double(beta_array,i,j);
        //                 swap_int(beta_index_array,i,j);
        //                 swap_double(IAT_array,i,j);
        //                 }
        //         }
        //     }
        // }

        for (int i=0; i<CONFIG; i++){
            int beta_i_index=-1;
            for (int k=0; k<CONFIG;k++){
                if (i==beta_index_array[k]){
                    beta_i_index=k;
                    break;
                } 
            }
            double x=action_array[beta_i_index];
            Thermfile<<x;
            if (i<CONFIG-1) Thermfile<< ",";
        }
        Thermfile<<"\n";
    }

    save_lattice_config(links,"Thermalized_Lattice_"+ std::to_string(SIZE)); 
    save_PT_Beta_Array(beta_array,"Beta_array");
    save_PT_index_Array(beta_index_array,"Beta_index_array");

    // Autocorrelation Procedure 

    for (int i=0; i<CONFIG;i++){
        for (int j=0; j<autocorrelation_sweeps;j++){
            double beta=beta_array[i];
            double action=update_heatbath(links, beta,i);
            autocorrelation_array[j]=action;
        }
        rho=autocorr(autocorrelation_array);
        double taucomp=tau_int(rho);
        if (taucomp < 1.0) taucomp = 1.0; 
        int tau=ceil(taucomp);
        IAT_array[i]=tau;
        std::cout<<"Beta "<< beta_array[i]<< " IAT "<< tau<< "\n"<<std::endl;
       }


    //Measurment Sweeps

    for (int j=0; j<measurment_sweeps;j++){
        for (int i=0; i<CONFIG;i++){
            double beta=beta_array[i];
            thermalize(links, IAT_array[i]-1,beta,i);
            double action=update_heatbath(links,beta,i);
            action_array[i]=action;   
        }

        for (int i = 0; i < CONFIG - 1; i++) {
            double delta = volume * (beta_array[i] - beta_array[i + 1]) *(action_array[i + 1] - action_array[i]);
            double Pswap = exp(delta);
            if (Pswap > 1.0) Pswap = 1.0;
            double x = sample_uniform();
            if (x < Pswap) {
                swap_double(beta_array, i, i + 1);
                swap_int(beta_index_array, i, i + 1);
                swap_double(IAT_array,i,i+1);
                }
            }

        for (int i=0; i<CONFIG; i++){
            int beta_i_index=-1;
            for (int k=0; k<CONFIG;k++){
                if (i==beta_index_array[k]){    
                    beta_i_index=k;
                    break;
                    }   
                }
            double x=action_array[beta_i_index];
            MCfile<<x;
            if (i<CONFIG-1) MCfile<<",";
            }
        MCfile<<"\n";
        }
    MCfile.close();
    Thermfile.close();
    return 0;
}




