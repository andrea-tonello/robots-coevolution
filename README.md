# Co-evolution of Robots with Genetic Programming

This project implements a co-evolutionary system where two populations of robots compete against each other in a 2D arena. The robots' strategies are evolved using Genetic Programming (GP), with each individual representing a decision-making program that processes sensor inputs to select actions.

## Features

- **Robot Simulation**: 2D arena with robots that can move, turn, shoot, reload, or do nothing.
- **Sensor System**: Robots perceive:
  - Enemy distance and direction
  - Wall proximity
  - Own health and ammo status
- **Genetic Programming**: Co-evolution of two competing populations using:
  - Tournament selection
  - Subtree crossover and mutation
  - Various primitive configurations
- **Analysis Tools**: 
  - Fitness tracking across generations
  - GP tree visualization
  - Comparative performance analysis

## Requirements

- Python 3.7+
- Required packages:
  - `deap`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `tqdm`
  - `graphviz`

Install requirements with:
```bash
pip install deap numpy pandas matplotlib tqdm graphviz
