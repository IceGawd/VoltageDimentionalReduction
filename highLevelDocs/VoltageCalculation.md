# Resistance Network Voltage Solver Documentation

## Overview

This document describes the mathematical formulation and implementation of a voltage solver over a resistance network. The core idea is to model a graph of points where some nodes called "landmarks" are held at fixed voltages. The remaining unconstrained node voltages are computed by solving a linear system that follows Kirchoff's Law.

## Problem Setup

Kirkchoff's Law states that the sum of currents entering a node has to be the same as the sum of currents leaving the node. In our case, each "node" is a data point or kmeans center of a data point. 

## calcResistanceMatrix

First thing that is done is the calculation of the resistance matrix. This name is a bit misleading, as we actually generate a conductivity matrix which has all of 

```
nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X)
```

