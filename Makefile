all: classical euclidean

classical: classical.py
	mv classical.py classical.py
euclidean: euclidean.c
	gcc euclidean.c -o euclidean -lm
