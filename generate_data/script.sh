python grid.py --output ../data/synthetic/1dgrid10.csv --grid_size 10 --d 1 --mode grid
python grid.py --output ../data/synthetic/2dgrid10.csv --grid_size 10 --d 2 --mode grid

python grid.py --output ../data/synthetic/1drandom10.csv --num_points  10 --d 1 --mode random
python grid.py --output ../data/synthetic/2drandom100.csv --num_points  100 --d 2 --mode random
python grid.py --output ../data/synthetic/2drandom10000.csv --num_points  10000 --d 2 --mode random
