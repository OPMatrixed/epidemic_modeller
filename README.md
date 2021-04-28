# Spatial Epidemic Modeller
## Alex Gardner's MA4J5 project

In this project we attempt to create a new epidemic modelling technique, similar to both Gillespie and tau-leaping, but this model simulates individuals moving in space.

An infected individual can only infect people in a given radius. We can also specify distributions for infection duration, latency period (in SEIR), and the time betwen infection attempts.

In this project I created a Python program to simulate an SEIR epidemic, and also allows you to render an animation of the individuals moving around the space, and seeing when/where people get infected.

## More details
Understanding this project assumes you have some basic understanding of epidemiology. If you want to see the project explained more in-depth, see Epidemic Modeller Overview.pdf.

There is also this video explaining and demonstrating the code: https://youtu.be/YRBOYNO0fDQ

### Running the demo
To run the examples notebook you need to have the requirements in requirements.txt satisfied. On my machine I have a venv with the required modules and jupyterlab.
