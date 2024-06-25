import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def EWA(datapoint, b):
    V = 0
    i = 0
    V_vec = [] # Initializing list, to then plot the EWA
    for datapoint in data:
        i += 1 # Increasing Time Step 
        V = b * V + ( 1 - b ) * datapoint # Computing EWA for time step t
        V = V / ( 1 - ( b ** i)) # Ridding of bias in EWA
        V_vec.append(V)

    return V_vec

       
if __name__ == "__main__":
    data = pd.read_csv('datasets/EWA.csv')
    data = np.array(data)
    b = 0

    V_vec = EWA(data, b)
    
    plt.plot(V_vec, label = 'EWA')
    plt.plot(data, label = 'Data')
    plt.legend()
    plt.show()

