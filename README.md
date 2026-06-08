# DART_DisTomo
Discrete tomographic reconstruction techniques DART, SDART and SSIRT, implemented using sklearn and the astra toolbox.
Requires a functional Nvidia GPU to use, as the ASTRA implementations only use CUDA.

## Required packages
Install required packages through `python-venv` or `uv`. `uv` is easier to use so that is recommended.

These instructions are for linux only, specifically debian.
### uv
1. Install `uv`:
- `curl -LsSf https://astral.sh/uv/install.sh | sh`
- uv is also availble on PyPi, so `pipx` and `pip` also works:
`pip install uv`
2. Installing packages:
- If you do not have any disk quotas, simply use:
`uv sync`
- If you DO have disk quotas, set a cache location for packages first:
`export UV_CACHE_DIR=/etc/data_location_here/.cache/uv/`
3. Running python
- Run scripts using:
`uv run python <script.py>` or `uv run python -m source.<script>` for module packages.
- Alternatively:
`source .venv/bin/activate` followed by
`python3 <script.py>` or `python3 -m source.<script>` for module packages.
### venv + pip
1. Create a .venv:
`python3 -m venv .venv`
2. Activate the .venv:
`source .venv/bin/activate`
3. Install packages:
`pip install -r requirements.txt`


## Before running
Before running any experiments, generate the phantom images using:

- `uv run python -m source.phantoms.create_phantoms` or `python3 -m source.phantoms.create_phantoms`

## Running the experiments
All experiment routines can be found in `./source/experiments/`, and have to be run as modules:


- `./source/experiments/lambda_sdart.py` is the routine for the lambda smoothing parameter for SDART.
    
    - To run:
    `uv run python -m source.experiments.lambda_sdart`

    - The results will be saved in `./lambda_results.json`.

- `./source/experiments/p_dart.py` is the routine for the free_pixels sampling chance for DART.

    - To run:
    `uv run python -m source.experiments.p_dart`

- `./source/experiments/visualize_algs.py` is a routine for creating some images for each algorithm of the phantom `"./phantoms/meshes/mesh_7.png"` for four different DetectorSuperSampling `a` parameters. The number of detectors for sinogram sampling is set to `64` with `180` projection angles. These will be stored in `./visuals/` as PNGs.

    - To run:
    `uv run python -m source.experiments.visualize_algs`

- `./source/experiments/run_all.py` is the script for running the complete experiments for all phantom images present in the `./phantoms/` directory. It will go through four `a` parameters for DetectorSuperSampling, `1, 4, 8` and `16` and take the mean `rNMP` and `SSIM` for each phantom family/group for each algorithm seperately. The sinograms are sampled using `64` detector elements, with the DetectorSuperSampling option during the sinogram sampling set to the same value as the detector element spacing. In the case of our experiments, this was `8`, as this makes the detector the same width as the phantoms, which was `512`. The script can be given two commandline options to change whether random poisson noise will be added to the sinogram or not.

    - To run using the original sinograms:
    `uv run python -m source.run_all`

        This will run all algorithms with clean sinograms.
    - To run using sinograms with added noise:
    `uv run python -m source.run_all noisy`

        This will add poisson noise using `astra.functions.add_noise_to_sino` with a background intensity of `1e5`.


## Phantoms
Here are some examples of what the phantoms will generally look like:
These are in order: "blob", "bone" and "mesh" phantom.

<img src="source/assets/example_blob.png" alt="Example 'blob' phantom" width="250"/> <img src="source/assets/example_bone.png" alt="Example 'bone' phantom" width="250"/> <img src="source/assets/example_mesh.png" alt="Example 'mesh' phantom" width="250"/>