### README
Code for generating sinograms and their SIRT and FBP reconstruction.

### Before running:
The original phantoms have to be placed in a directory named "phantoms/". The actual
PNGs can have any name as long as they are placed in working_dir/phantoms/.
This is automatically done by the previous phantoms script "phantom_generator.py".

### Required packages:
- Numpy
- PIL
- astra-toolbox

### Running
1. create a venv or conda env with python3.12
2. activate environment
3. install required packages with pip install -r requirements.txt
4. run the script:
- python3 sinograms.py
