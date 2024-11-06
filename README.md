# Datasets for "Learning out-of-time-ordered correlators with classical kernel methods"

This repository contains all datasets used in the paper *"Learning out-of-time-ordered correlators with classical kernel methods"* (Authors: John Tanner, Jason Pye, Jingbo Wang; Year: 2024; arXiv: 2409.01592). The datasets are made available for research and reproducibility purposes. Below you will find descriptions of how the datasets are named, how to access and use them, and other relevant information. 

We kindly request users who wish to use the datasets in their own research to cite the corresponding paper.

## Data Naming Convention

In total there are 128 datasets in the repository whose names are formatted as:

*"\<**Hams**\>_\<**Func**\>__n\<**N_qubits**\>_chi\<**Max_bond_dim**\>_tmax\<**Max_ev_time**\>_dt\<**Trotter_step**\>_samples\<**Num_data**\>.csv"*

where
- ***Hams*** = H1, H2, H3, H4: specifies which of the four parameterised sets of Hamiltonians is being used (defined in equations (18) - (21) of the paper),
- ***Func*** = XZOTOC, SumOfOTOCs: specifies whether we are calculating the XZ-OTOC or sum of OTOCs (defined in equations (22) and (23) of the paper),
- ***N_qubits*** = 5, 10, 15, ..., 40: specifies the size of the underlying quantum system being simulated,
- ***Max_bond_dim*** = 70, 90, 100, 110, 140, 150: specifies the maximum bond dimension used to calculate the associated function (Func) with the MPO-based algorithm (see Section III.D of the paper for more details),
- ***Max_ev_time*** = n, 2n: specifies the maximum time for which the underlying quantum system may be evolved under the associated Hamiltonian (see Section III.A and III.B of the paper for more details),
- ***Trotter_step*** = 0.05: specifies the step size used in the Trotterisation of the Hamiltonians during the TEBD used in the MPO-based algorithm (see Section III.D of the paper for more details),
- ***Num_data*** = 250, 1000: specifies how many data samples the dataset contains. In the paper, we use the datasets with Num_data=1000 for training, and those with Num_data=250 for testing.

## Data Format

- **File Format**: The datasets are provided in .csv format.
- **Column Descriptions**: The first, second, and third columns contain the first, second, and third components of the input data vector $x\in\mathbb{R}^3$ (see Section III.A of the paper for more details). The fourth column contains the associated value of either the XZ-OTOC or sum of OTOCs calculated with the MPO-based algorithm.

## Data Usage

- **Loading the Data**: To load the datasets with Python, you can use:
  - `pandas`:
    
    ```python
    import pandas as pd
    data = pd.read_csv('path_to_dataset.csv')
    data_inputs = data.iloc[:, :3]  
    data_labels = data.iloc[:, 3] 
    ```
  - `numpy`:
    
    ```python
    import numpy as np
    data = np.genfromtxt('path_to_dataset.csv', delimiter=',', skip_header=1)
    data_inputs = data[:, :3]
    data_labels = data[:, 3]
    ```

- **Visualising the data**: To visualise the datasets with Python, you can use:
  - `pandas` and `matplotlib`:
    
    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    data = pd.read_csv('path_to_dataset.csv')
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    image = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], c=data.iloc[:, 3], s=75)
    plt.colorbar(image)
    plt.show()
    ```

  - `numpy` and `matplotlib`:
    
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    data = np.genfromtxt('path_to_dataset.csv', delimiter=',', skip_header=1)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    image = ax.scatter3D(data[:,0],data[:,1],data[:,2],c=data[:,3],s=75)
    plt.colorbar(image)
    plt.show()
    ```

## Contact

For any questions, issues, or feedback, please send an email to: john.tanner@uwa.edu.au



