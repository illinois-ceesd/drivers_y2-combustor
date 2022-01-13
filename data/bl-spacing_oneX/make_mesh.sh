#!/bin/bash
gmsh -setnumber size 0.0008 -setnumber blratio 4 -o combustor.msh -nopopup -format msh2 ./combustor.geo -2
