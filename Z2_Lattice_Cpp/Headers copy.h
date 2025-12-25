#pragma once
#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <omp.h>
#include <fstream>
#include <algorithm>

extern std::mt19937 rng;

// Simulation Parameters

// Lattice Configurations 
constexpr int SIZE = 6;
constexpr int CONFIG = 50;

// Beta 
constexpr double dbeta=.00025;
constexpr double beta_min=0.43;


// Measurments
constexpr int thermal_sweeps=1000000;  
constexpr int autocorrelation_sweeps=10000;  
constexpr int measurment_sweeps=100000;  
constexpr int maxlag=8000;




// Constants to not be manipulated
constexpr int array_size = 4*SIZE*SIZE*SIZE*SIZE*CONFIG;
constexpr int volume = SIZE*SIZE*SIZE*SIZE*6;
constexpr double beta_max=beta_min+CONFIG*dbeta;



// RNG
double sample_uniform(std::mt19937 &rng_local);
std::array<int,4> sample_coord(std::mt19937 &rng_local);


// Lattice configuration updates
void cold_start_array(std::array <int, array_size>& arr);
void moveup(std::array<int,4>& v,int d);
void movedown(std::array<int,4>& v, int d);
double return_action(std::array<int,array_size>& link, int lattice_config);


// Save Config
using MyArray = std::array<int, array_size>;
void save_lattice_config(const MyArray& array, const std::string& filename); 
void save_PT_Beta_Array(const std::array<double,CONFIG>& array, const std::string& filename);
void save_PT_index_Array(const std::array<int,CONFIG>& array, const std::string& filename);
std::array<double, CONFIG> load_PT_Beta_Array(const std::string& filename);
std::array<int, CONFIG> load_PT_index_Array(const std::string& filename);
MyArray load_lattice_config(const std::string& filename);

// Swap elements
void swap_double(std::array<double,CONFIG>& v, int index1, int index2);
void swap_int(std::array<int,CONFIG>& v, int index1, int index2);

// Monte Carlo updates
double update_heatbath(std::array<int,array_size>& link, double beta, int lattice_config,std::mt19937 &rng_local);
double update_metropolis(std::array<int,array_size>& link, double beta, int lattice_config,std::mt19937 &rng_local);


void thermalize(std::array<int,array_size>& link, int therm_sweeps, double beta, int lattice_config,std::mt19937 &rng_local);



// Overrelaxation
void gauge_transformation(std::array<int, array_size>& link, int lattice_config,std::array<int,4> x);
void overrelaxation(std::array<int,array_size>& link, int lattice_config);
void flip_lattice_links(std::array<int,array_size>& link);

// Autocorrelation analysis
std::vector<double> autocorr(const std::vector<double>& data, int max_lag);
float tau_int(const std::vector<double>& autocorr_data, int maxlag);

// Tensor analysis
int flat_index(int i1, int i2, int i3, int i4, int i5, int i6);
int get_array_value(const std::array<int,array_size>& arr, int i1, int i2, int i3, int i4, int i5, int i6) ;
void set_array_value(std::array<int,array_size>& arr, int i1, int i2, int i3, int i4, int i5, int i6, int value);
std::array<int,6> tensor_index_array(int index);


// Statistics
double mean(const std::array<double,autocorrelation_sweeps>& data);
double variance(const std::array<double,autocorrelation_sweeps>& data);
std::array<double,maxlag> autocorr(const std::array<double,autocorrelation_sweeps>& data);
double tau_int(const std::array<double,maxlag>& rho);