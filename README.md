## connectivity: a multi-resolution landscape connectivity algorithm

A multi-resolution landscape connectivity algorithm for calculating *Habitat Condition Connectedness* and the *Bioclimatic Ecosystem Resilience Index (BERI)*.

This algorithm operates on the overview layers of Cloud Optimized GeoTIFF (COG) files. Please ensure that these overview layers are generated using `mean` aggregation, not `nearest neighbor` resampling. Use `create_overviews()` function for generting correct overview layers.

### Installation

You need to load the module and create an environment if you don't alreay have one.

```bash
module load python/3.12.3
module load rust/1.84.1
```

```bash
python -m venv ~/myenv
```
Installing the library into the environent:

```bash
source ~/myenv/bin/activate

# To make the virtual environment explicitly include system site packages, use:
python -m venv ~/myenv --system-site-packages
```

Build a wheel and install it:

```bash
cd ~/connectivity

maturin build --release
```
Then find the wheel under `target/wheels/`, and install it with:

```bash
pip install target/wheels/connectivity-*.whl
```

### Example

```python
from connectivity import connectedness, beri, create_overviews, overview_info
```

**1. Inspecting Overview Information**    
You can inspect the overview metadata of a TIFF file using the overview_info() function:

```python
overview_info(grids)
```
```
File: ../data/transgrids/1990.tif
Resolution: 588 x 516
Bands: 19
CRS: EPSG:4326

Overview Information:
  Band 1 overviews: [1, 2, 4, 8, 16, 32]
  Overview resolutions:
    Level 1: 588 x 516
    Level 2: 294 x 258
    Level 4: 147 x 129
    Level 8: 73 x 64
    Level 16: 36 x 32
    Level 32: 18 x 16
```

**Missing Overviews:**    
If a file has no existing overviews, the output will indicate an empty list:

```
Resolution: 588 x 516
Bands: 19
CRS: EPSG:4326

Overview Information:
  Band 1 overviews: []
```

**2. Creating Overviews**     
Use the `create_overviews()` function to generate and embed overviews into the TIFF file.

```python
create_overviews(
    input_raster = "file.tif",
    output_raster = None, # keep None to update layer inplace
    overview_levels = [1, 2, 4, 8, 16] # the overview levels
)
```

**3. Calculating connectivity or BERI**    

```python
connd = connectedness(
    condition_file = "cog_file.tif",
    lambdas = [2.0, 20.0, 200.0],
    scale = 2.0, 
    nb_size = 3, 
    last_nb_size = 5,
    levels = [1, 2, 4, 8, 16],
    mask = True, 
    sigma = 5,
    filename = ""
)

```