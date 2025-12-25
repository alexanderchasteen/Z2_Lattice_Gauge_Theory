#include "Headers copy.h"

// Save MC Config

// Saves the lattice Configuration at the end of a thermal cycle if not long enough
void save_lattice_config(const MyArray& array, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    out.write(reinterpret_cast<const char*>(array.data()),
              array.size() * sizeof(int));
}


// Save the beta array used in tempering
void save_PT_Beta_Array(const std::array<double,CONFIG>& array, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    out.write(reinterpret_cast<const char*>(array.data()),
              array.size() * sizeof(double));
}


// Save the beta index array used in tempering
void save_PT_index_Array(const std::array<int,CONFIG>& array, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    out.write(reinterpret_cast<const char*>(array.data()),
              array.size() * sizeof(int));
}


// load the saved configurations into an array
MyArray load_lattice_config(const std::string& filename) {
    MyArray array{}; // default-initialize to zero
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    in.read(reinterpret_cast<char*>(array.data()), array.size() * sizeof(int));
    if (!in) {
        throw std::runtime_error("Error reading file: " + filename);
    }

    return array;
}

// same
std::array<double, CONFIG> load_PT_Beta_Array(const std::string& filename) {
    std::array<double, CONFIG> array{};
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    in.read(reinterpret_cast<char*>(array.data()), array.size() * sizeof(double));
    if (!in) {
        throw std::runtime_error("Error reading file: " + filename);
    }
    return array;
}

// Load int array (PT_index_Array)
std::array<int, CONFIG> load_PT_index_Array(const std::string& filename) {
    std::array<int, CONFIG> array{};
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    in.read(reinterpret_cast<char*>(array.data()), array.size() * sizeof(int));
    if (!in) {
        throw std::runtime_error("Error reading file: " + filename);
    }
    return array;
}



// Initialize RNG for the update function
std::mt19937 rng(std::random_device{}());

double sample_uniform(std::mt19937 &rng_local) {
    static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_local);
}

std::array<int,4> sample_coord(std::mt19937 &rng_local) {
    static thread_local std::uniform_int_distribution<int> dist2(0, SIZE-1);
    int x0 = dist2(rng_local);
    int x1 = dist2(rng_local);
    int x2 = dist2(rng_local);
    int x3 = dist2(rng_local);
    return {x0,x1,x2,x3};
}

// Gives the index in an array given the indices in our 6 index tensor for the lattice_config x lattice_data information 
int flat_index(int i1, int i2, int i3, int i4, int i5, int i6) {
    return (((((i1 * SIZE + i2) * SIZE + i3) * SIZE + i4) * SIZE + i5) * 4 + i6);
}

// gives the 6 indices in the tensor given the location in the array
std::array<int,6> tensor_index_array(int index){
    int i6= index % 4;
    index /=4;
    int i5 = index % SIZE;
    index /= SIZE;
    int i4 = index % SIZE;
    index /= SIZE;
    int i3 = index % SIZE;
    index /= SIZE;
    int i2 = index % SIZE;
    index /= SIZE;
    int i1 = index;
    return {i1,i2,i3,i4,i5,i6};
}

// Change the value at a certain point in the array
void set_array_value(std::array<int,array_size>& arr, int i1, int i2, int i3, int i4, int i5, int i6, int value) {
    int idx = flat_index(i1, i2, i3, i4, i5, i6);
    arr[idx] = value;
    return;
}

// Gets the value at a certain point in the array
int get_array_value(const std::array<int,array_size>& arr, int i1, int i2, int i3, int i4, int i5, int i6) {
    int idx = flat_index(i1, i2, i3, i4, i5, i6);
    return arr[idx];
}

// Sets all the links to 1. Same as iteratring through the array but this way is to make sure consistint with the array construction
void cold_start_array(std::array <int, array_size>& arr){
    for (int i1 = 0; i1 < CONFIG; i1++)
    for (int i2 = 0; i2 < SIZE; i2++)
    for (int i3 = 0; i3 < SIZE; i3++)
    for (int i4 = 0; i4 < SIZE; i4++)
    for (int i5 = 0; i5 < SIZE; i5++)
    for (int i6 = 0; i6 < 4; i6++)
    set_array_value(arr,i1,i2,i3,i4,i5,i6,1);
    return;
    }


// For a 4 index array representing the lattice points this allows you to move in a positive direction in the lattice to a new node
void moveup(std::array<int,4>& v,int d){
    v[d]+=1;
    if (v[d]>=SIZE) v[d]-=SIZE;
    return;
}

// Same as moveup but in negative direction
void movedown(std::array<int,4>& v,int d){
    if (v[d]==0) v[d]+=SIZE-1;
    else v[d]-=1;
    return;
}

// Swaps elements of an array that are doubles
void swap_double(std::array<double,CONFIG>& arr, int index1, int index2){
    double tmp=arr[index1];
    arr[index1]=arr[index2];
    arr[index2]=tmp;
}


// Swaps elements of an array that are non-neg ints (int)
void swap_int(std::array<int,CONFIG>& arr, int index1, int index2){
    int tmp=arr[index1];
    arr[index1]=arr[index2];
    arr[index2]=tmp;
}

// Updates one lattice config in the link array according to some float beta=1/kT
double update_metropolis(std::array<int,array_size>& link, double beta, int lattice_config,std::mt19937 &rng_local){
    std::array<int, 4> x; 
    int d,current,dperp,staple,staplesum;
    double metropolis_number,action=0.0;
    for (x[0]=0; x[0]<SIZE; x[0]++)
        for (x[1]=0; x[1]<SIZE; x[1]++)
            for (x[2]=0; x[2]<SIZE; x[2]++)
                for (x[3]=0; x[3]<SIZE; x[3]++)
                    for (d=0; d<4; d++) {
                        staplesum=0;
                        for (dperp=0;dperp<4;dperp++){
                            if (dperp!=d){
                                std::array<int, 4> y=x; 
                                movedown(y,dperp);
                                staple=get_array_value(link,lattice_config,y[0],y[1],y[2],y[3],dperp);
                                staple*=get_array_value(link,lattice_config,y[0],y[1],y[2],y[3],d);
                                moveup(y,d);
                                staple*=get_array_value(link,lattice_config,y[0],y[1],y[2],y[3],dperp);
                                moveup(y,dperp);
                                staplesum+=staple;
                                staple=get_array_value(link,lattice_config,y[0],y[1],y[2],y[3],dperp);
                                moveup(y,dperp);
                                movedown(y,d);
                                staple*=get_array_value(link,lattice_config,y[0],y[1],y[2],y[3],d);
                                movedown(y,dperp);
                                staple*=get_array_value(link,lattice_config,y[0],y[1],y[2],y[3],dperp);
                                staplesum+=staple;
                            }
                        }


                        // Compute exp(-2beta|Staplesum|) and this will automatically be accepted by metroplis updatae
                        // Sample uniform and if less than then accept the change. Otherwise do not. 
                        
                        current=get_array_value(link,lattice_config,x[0],x[1],x[2],x[3],d);
                        metropolis_number=std::min(std::exp(-2*beta*staplesum*current),1.0);
                        
                        
                        if (sample_uniform(rng_local) < metropolis_number){
                            set_array_value(link,lattice_config,x[0],x[1],x[2],x[3],d,-current);
                            action+=-staplesum*current;
                        }
                        else {
                            set_array_value(link,lattice_config,x[0],x[1],x[2],x[3],d,current);
                            action+=staplesum*current;
                        }
                    
                    }
    action/=(SIZE*SIZE*SIZE*SIZE*4*6);
    return action;
}

double update_heatbath(std::array<int,array_size>& link, double beta, int lattice_config, std::mt19937 &rng_local){
    std::array<int, 4> x; 
    int d,dperp,staple,staplesum;
    double bplus,bminus,action=0.0;
    for (x[0]=0; x[0]<SIZE; x[0]++)
        for (x[1]=0; x[1]<SIZE; x[1]++)
            for (x[2]=0; x[2]<SIZE; x[2]++)
                for (x[3]=0; x[3]<SIZE; x[3]++)
                    for (d=0; d<4; d++) {
                        staplesum=0;
                        for (dperp=0;dperp<4;dperp++){
                            if (dperp!=d){
                                std::array<int, 4> y=x; 
                                movedown(y,dperp);
                                staple=get_array_value(link,lattice_config,y[0],y[1],y[2],y[3],dperp);
                                staple*=get_array_value(link,lattice_config,y[0],y[1],y[2],y[3],d);
                                moveup(y,d);
                                staple*=get_array_value(link,lattice_config,y[0],y[1],y[2],y[3],dperp);
                                moveup(y,dperp);
                                staplesum+=staple;
                                staple=get_array_value(link,lattice_config,y[0],y[1],y[2],y[3],dperp);
                                moveup(y,dperp);
                                movedown(y,d);
                                staple*=get_array_value(link,lattice_config,y[0],y[1],y[2],y[3],d);
                                movedown(y,dperp);
                                staple*=get_array_value(link,lattice_config,y[0],y[1],y[2],y[3],dperp);
                                staplesum+=staple;
                            }
                        }
                        
            /* the heatbath algorithm */
                        bplus=std::exp(beta*staplesum);
                        bminus=1/bplus;
                        bplus=bplus/(bplus+bminus);

                        if (sample_uniform(rng_local) < bplus){
                            set_array_value(link,lattice_config,x[0],x[1],x[2],x[3],d,1);
                            action+=staplesum;
                        }
                        else {
                            set_array_value(link,lattice_config,x[0],x[1],x[2],x[3],d,-1);
                            action-=staplesum;
                        }
                                              

                    }
    action/=(SIZE*SIZE*SIZE*SIZE*4*6);
    return action;
}


double return_action(std::array<int,array_size>& link, int lattice_config){
    std::array<int, 4> x; 
    int d,dperp,staple,staplesum;
    double action=0.0;
   for (x[0]=0; x[0]<SIZE; x[0]++)
        for (x[1]=0; x[1]<SIZE; x[1]++)
            for (x[2]=0; x[2]<SIZE; x[2]++)
                for (x[3]=0; x[3]<SIZE; x[3]++)
                    for (d=0; d<4; d++) {
                        staplesum=0;
                        for (dperp=0;dperp<4;dperp++){
                            if (dperp!=d){
                                std::array<int, 4> y=x; 
                                movedown(y,dperp);
                                staple=get_array_value(link,lattice_config,y[0],y[1],y[2],y[3],dperp);
                                staple*=get_array_value(link,lattice_config,y[0],y[1],y[2],y[3],d);
                                moveup(y,d);
                                staple*=get_array_value(link,lattice_config,y[0],y[1],y[2],y[3],dperp);
                                moveup(y,dperp);
                                staplesum+=staple;
                                staple=get_array_value(link,lattice_config,y[0],y[1],y[2],y[3],dperp);
                                moveup(y,dperp);
                                movedown(y,d);
                                staple*=get_array_value(link,lattice_config,y[0],y[1],y[2],y[3],d);
                                movedown(y,dperp);
                                staple*=get_array_value(link,lattice_config,y[0],y[1],y[2],y[3],dperp);
                                staplesum+=staple;
                            }
                        }
                        int main_link=get_array_value(link,lattice_config,x[0],x[1],x[2],x[3],d);
                        action+= main_link*staplesum;
                    }
    action/=(SIZE*SIZE*SIZE*SIZE*4*6);
    return action;
}



void thermalize(std::array<int,array_size>& link, int therm_sweeps, double beta, int lattice_config, std::mt19937 &rng_local){
    for (int i=0; i<therm_sweeps; i++){
        update_heatbath(link,beta,lattice_config,rng_local);
    }
}


void flip_lattice_links(std::array<int,array_size>& link){
    for (int i = 0; i < array_size; i++) {
        link[i] = -link[i];  // flip link
    }
}


// Autocorrelation Functions
double mean(const std::array<double,autocorrelation_sweeps>& data) {
    double sum=0.0;
    for (double x : data) sum += x;
    return sum / data.size();
}

double variance(const std::array<double,autocorrelation_sweeps>& data){
    double m=mean(data);
    double sum=0.0;
    for (double x : data){
        double diff= x-m;
        sum+= diff*diff;
    }
    return sum / data.size();
}


std::array<double,maxlag> autocorr(const std::array<double,autocorrelation_sweeps>& data) {
    std::array<double,maxlag> rho {};
    double m=mean(data);
    double var=variance(data);
    int N=data.size();
    if (var == 0.0){
        for (int t=0; t<maxlag; t++) rho[t] = (t==0)?1.0:0.0;
        return rho;
    }
    for (int t = 0; t < maxlag; t++){
        double c = 0.0;
         for (int i = 0; i < N - t; i++) {
            c += (data[i] - m) * (data[i + t] - m);
        }
        c /= (N- t);   
        rho[t] = c / var;
    }
    return rho;
}

double tau_int(const std::array<double,maxlag>& rho) {
    double tau = 1;
    for (int t = 1; t < maxlag; t++) {
     tau += 2*  rho[t];
    }
    return tau;
}


