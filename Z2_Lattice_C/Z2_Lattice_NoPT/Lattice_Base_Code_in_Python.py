import numpy as np
import math

# setting lattice dimensions
SIZE = 3
lattice = np.ones((SIZE, SIZE, SIZE, SIZE, 4), dtype=np.int8)
#sets the lattice in a cold start



# Defining Utlity Functions
def moveup(x,d):
    x[d]+=1
    if x[d]>=SIZE:
        x[d]-=SIZE
    return

def movedown(x,d):
    x[d]-=1
    if x[d]<0:
        x[d]+=SIZE
    return


# Use same random function maybe--can switch back to np.random()
class DRand48:
    def __init__(self, seed=1234):
        # initialize like srand48(seed)
        self.state = (seed << 16) + 0x330E
        self.a = 0x5DEECE66D
        self.c = 0xB
        self.m = 1 << 48

    def drand48(self):
        self.state = (self.a * self.state + self.c) % self.m
        return self.state / self.m



def update(beta,rng):
    action=0.0
    for x0 in range(SIZE):    
        for x1 in range(SIZE):  
            for x2 in range(SIZE):  
                for x3 in range(SIZE):
                    for d in range(4):
                        staplesum=0.0 
                        x=[x0,x1,x2,x3]
                        #stample_sum collects the sum of the action contributions from neighboring plaqs touching link in direction d
                        #There will be 2 plaqs in each perpendicular direction, so 6 sums total
                        #This will give the local action at the link. The proability of this link update is dependent only the boltzman factors is local
                        #This is because probability that link updates given the rest of the configuration splits into the rest of the lattice plus this part that depends Only on the lattice
                        #so we compute staple sum and RNG wrt. boltzman facttors to update
                        for dperp in range(4):
                            if dperp != d:  
                                movedown(x,dperp) 
                                staple=lattice[x[0]][x[1]][x[2]][x[3]][dperp]*lattice[x[0]][x[1]][x[2]][x[3]][d]
                                moveup(x,d)
                                staple*=lattice[x[0]][x[1]][x[2]][x[3]][dperp]    
                                moveup(x,dperp)
                                staplesum+=staple
                                #first plaq done; now second
                                staple=lattice[x[0]][x[1]][x[2]][x[3]][dperp]
                                moveup(x,dperp)
                                movedown(x,d)
                                staple*=lattice[x[0]][x[1]][x[2]][x[3]][d]
                                movedown(x,dperp)
                                staple*=lattice[x[0]][x[1]][x[2]][x[3]][dperp]
                                staplesum+=staple
                        bplus=np.exp(beta*staplesum)
                        bminus=1/bplus
                        bplus=bplus/(bplus+bminus)
                        if rng.drand48() < bplus:
                            lattice[x0, x1, x2, x3, d] = 1
                            action += staplesum

                        else:
                            lattice[x0, x1, x2, x3, d] = -1
                            action -= staplesum
    action /= (SIZE * SIZE * SIZE * SIZE * 4 * 6)                           
    return action

def thermalize(thermalize_sweeps,beta):
    for i in range(thermalize_sweeps):
        update(beta)

def main():
    rng = DRand48(seed=1234)
    np.random.seed(1234)
    dbeta = 0.01
    thermal_sweeps=100
    # heat it up
    beta = 1.0
    while beta > 0.0:
        #thermalize(thermal_sweeps,beta) 
        action = update(beta,rng)
        print(f"{action:.6f}")
        beta -= dbeta
    print("\n")


main()