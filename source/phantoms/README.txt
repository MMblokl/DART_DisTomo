# README
Code for generating 3 phantom images according to the set seed.
Phantoms are saved in phantoms/*.png

# Required packages:
- Numpy
- PIL
- matplotlib
- scikit-image

# Running
1. create a venv or conda env with python 3.10 or 3.12:
- Venv: python3 -m venv env_name
- Conda: create --name env_name python=3.12

2. activate environment:
- Venv: source env_name/bin/activate
- Conda: conda activate env_name

3. install required packages:
- pip install -r requirements.txt

4. run the script:
- python3 phantom_generator.py
