#!/bin/bash


cat $* | python -c 'import sys; print(len(set( sys.stdin.readlines())));'
