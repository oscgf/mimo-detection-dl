# MIMO Detection with Deep Learning

## Summary

This project investigates the use of Deep Learning techniques for detection in Multiple-Input Multiple-Output (MIMO) communication systems. It is part of the Final Degree Project in Telecommunications Engineering, ***MIMO Detection with Deep Learning***, conducted by Óscar González Fresno, under the supervision of Juan José Murillo Fuentes, Professor at the University of Seville.

The main objective is to **improve the efficiency of the Maximum Likelihood (ML) detection method, which, although optimal, presents a high computational burden**, especially as the number of antennas and constellation points increases.

The project is divided into three parts:

1. Simulation of the MIMO System in Python
2. Simulation of the MIMO System in PyTorch
3. Design and simulation of MIMO detection based on deep learning

You can download and run the files in a Python environment like Anaconda, though it is recommended to execute the notebooks in Google Colab, taking advantage of hardware acceleration and loading the files _labcomgid.py_, _functions.py_, and _network.py_.

An environment file is also included.

## Key Features

* **MIMO communication system simulation:** A MIMO system is implemented in Python using the NumPy library, including signal generation, AWGN noise addition, and detection.
* **Comparison of detection methods:** Three detection methods are evaluated: Zero Forcing (ZF), Linear Minimum Mean Squared Error (LMMSE), and Maximum Likelihood (ML).
* **Implementation with PyTorch:** The MIMO system is migrated to PyTorch to leverage GPU processing power and tensor operations.
* **Design of a neural network for ML detection:** A neural network is designed and implemented using PyTorch to replicate the behavior of the ML detector, aiming to reduce computational complexity.

## Results

* **The traditional ML method shows better performance in terms of Bit Error Rate (BER) compared to ZF and LMMSE, especially in channels with poorly conditioned matrices.**
* **The PyTorch implementation significantly accelerates the detection process for ZF and LMMSE but does not show significant improvements for ML.**
* **The trained neural network achieves a BER similar to traditional ML for low SNR values. However, its performance deteriorates as SNR increases.**

## Future Research Directions

* **Optimization of the traditional ML detection algorithm to reduce its computational burden.**
* **Investigation into the performance difference of the ML algorithm between NumPy and PyTorch.**
* **Design of more complex neural networks to handle MIMO systems with a larger number of antennas and modulation points.**

## Dependencies

* Python 3
* NumPy
* Matplotlib
* PyTorch

## Content

* _enviroment.yml_ - Programming environment file
* _MIMO_numpy.py_ - MIMO system that evaluates the ZF, LMMSE, and ML detection methods for
		         3 manually generated channel matrices.
* _Comparison.py_ - Simulation of the system in various channels with randomly generated coefficient matrices
			 with normal distribution.
* _mimo_pytorch.py_   	- _.py_ version of _MIMO_pytorch.ipynb_
* _mimo_nn.py_	   	- _.py_ version of _MIMO_pytorch.ipynb_

* _labcomgid.py_ 	- File containing functions for digital communications lab exercises.
* _functions.py_  	- File with new functions created for this project.
* _network.py_	   	- File defining several classes used in _MIMO_nn.ipynb_.

* _MIMO_pytorch.ipynb_ 	- Notebook with the PyTorch-based system simulating a
		         MIMO system with the three detection methods.
* _MIMO_nn.ipynb_	- Program using a neural network for symbol detection,
			 comparing it with the ZF, LMMSE, and ML methods.

## How to Run the Code

1. Clone the GitHub repository.
2. Install the necessary dependencies.
3. Run the `MIMO_numpy.py` file for the NumPy-based simulation.
4. Run the `MIMO_pytorch.ipynb` or `mimo_pytorch.py` file for the PyTorch-based simulation.
5. Run the `MIMO_pytorch.ipynb` or `mimo_nn.py` file for the neural network-based simulation.

## Contact

* Óscar González Fresno - oscargonzalezfresno@gmail.com

## Acknowledgements

* Thanks to the project supervisor, Juan José Murillo Fuentes, for his invaluable guidance and support. 
* Recognition to the Department of Signal Theory and Communications of the University of Seville for providing the foundational material for this project.

## References

[1] F. J. Payán Somet, J. J. Murillo Fuentes, and J. C. Aradillas Jaramillo, "Manual de Laboratorio de Comunicaciones Digitales con Python," 2020 ed., Departamento de Teoría de la Señal y Comunicaciones. Escuela Técnica Superior de Ingeniería, Universidad de Sevilla.

[2] "Anaconda," [Online]. Available: [https://www.anaconda.com/](https://www.anaconda.com/).

[3] "Spyder IDE," [Online]. Available: [https://www.spyder-ide.org/](https://www.spyder-ide.org/).

[4] "PyTorch," [Online]. Available: [https://pytorch.org/](https://pytorch.org/).

[5] "NumPy," [Online]. Available: [https://numpy.org/](https://numpy.org/).

[6] "Matplotlib," [Online]. Available: [https://matplotlib.org/](https://matplotlib.org/).

[7] Ó. González Fresno, "Github," February 2022. [Online]. Available: [https://github.com/osgofre/MIMO-DL](https://github.com/osgofre/MIMO-DL).

[8] "Additive White Gaussian Noise," Wikipedia, 25 October 2021. [Online]. Available: [https://en.wikipedia.org/wiki/Additive_white_Gaussian_noise](https://en.wikipedia.org/wiki/Additive_white_Gaussian_noise). [Last accessed: 11 January 2022].

[9] A. Goldsmith, "MIMO detection algorithms," in *Wireless Communications*, Stanford University, 2017.

[10] "Número de condición," Wikipedia, 12 January 2022. [Online]. Available: [https://es.wikipedia.org/wiki/N%C3%BAmero_de_condici%C3%B3n](https://es.wikipedia.org/wiki/N%C3%BAmero_de_condici%C3%B3n). [Last accessed: 12 January 2022].

[11] J. J. M. Fuentes, "Unit 3: Notes on the LMMSE in MIMO," in *Procesado Digital para Sistemas de Comunicaciones y Audiovisuales*, Departamento de Teoría de la Señal y Comunicaciones. Escuela Técnica Superior de Ingeniería, Universidad de Sevilla, 2021.

[12] "Google Colaboratory," Google, LLC, [Online]. Available: [https://colab.research.google.com/notebooks/welcome.ipynb?hl=es](https://colab.research.google.com/notebooks/welcome.ipynb?hl=es).

[13] R. Raina, A. Madhavan, and A. Y. Ng, *Large-scale Deep Unsupervised Learning using Graphics Processors*, ACM Press the 26th Annual International Conference - Montreal, Quebec, Canada (2009.06.14-2009.06.18), Stanford, 2009.

[14] Udacity, "Intro to Deep Learning with PyTorch," Facebook, 2021. [Online]. Available: [https://www.udacity.com/course/deep-learning-pytorch--ud188](https://www.udacity.com/course/deep-learning-pytorch--ud188).

[15] Y. LeCun, C. Cortes, and C. J.C. Burges, "The MNIST Database," [Online]. Available: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/).

[16] "Datasets & Dataloaders," PyTorch, [Online]. Available: [https://pytorch.org/tutorials/beginner/basics/data_tutorial.html](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

