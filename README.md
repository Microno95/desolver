# DESolver
The purpose of the integrator python script is to provide a straightforward interface for numerically integrating
systems of linear and non-linear first-order Ordinary Differential Equations, and plotting the results.
Several different integration methods have been implemented and several interfacing methods exist for using the script.

#To Install:
Just type

	pip install DESolver
or

	conda install DESolver

##Implemented Integration Methods
1. Explicit Runge-Kutta 4
2. Forward Euler
3. Backward Euler
4. Implicit Midpoint
5. Explicit Midpoint
6. Adaptive Heunn-Euler
7. Euler-Trapezoidal Method
8. Heun's Method
9. Symplectic Forward Euler

		NOTE:   The Symplectic Foward Euler method takes the entered equations in pairs for integration. For example,
						when integrating a system where the position is updated based on the intermediate future velocity
						then the equations would look something like the following for a spring satisfying Hooke's Law.
						Equation 1 = y_1/m
						Equation 2 = -k*y_0
						Where y_0 is the position, y_1 is the momentum, k is the spring stiffness, and m is the mass.


##Usage Methods
1. Directly Running the Script

	When you run the script without any command line/terminal arguments the script will automatically default to
	the internal interface where you will be asked how you wish to enter the various parameters for integration.

	This input method is the best documented and the directions are stated clearly when run.
	Basically the script will ask for various parameters, separated by commas (or semicolons in the case
	of the plotting methods), check if the parameters have been correctly entered, and request clarification or
	re-entering if necessary.

2. Parameters from a Text File

	When running the script, using the -t optional argument followed by the name and location of a text file with
	the integration parameters will run the script directly using the arguments provided.

	If you do set the script to make plots then it will request that you enter the pairs of variables to plot
	followed by requesting the necessarily titles and other parameters for a matplotlib 2d, line or scatter plot.

3. Parameters from the Command Line/Terminal
	
	**WARNING: This interface is still experimental and requires improvement.**

	When you run the script with the optional arguments:
	- -eqn "...ode 1..." "...ode 2..." ... "...ode n..."
	- -y_i y_0(0) y_1(0) ... y_n(0)
	- -tp t_initial t_final step_size
	- -o "output directory location"
	- -m "method of integration"
	
	In that order, the script will run and give the relevant output to the directory you have specified.


I hope that this script is useful to you. If you have any suggestions as to what I should implement next email me or
fork the repository and implement it yourself.

**DISCLAIMER:
The commenting on the code is currently sub-par and requires some polishing as certain variables and
functions are not described in sufficient detail. Please email me if you wish to understand the code
better.**
