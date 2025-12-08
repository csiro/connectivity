## connectivity: a multi-resolution landscape connectivity algorithm

A multi-resolution landscape connectivity algorithm for calculating *Habitat Connectedness (Connected-Habitat)*, *PARC Connectedness* and the *Bioclimatic Ecosystem Resilience Index (BERI)*.

This algorithm operates on the overview layers of a GeoTIFF file (including Cloud-Optimized GeoTIFFs). Please ensure that these overview layers are generated using the `average`, not `nearest` resampling method. Use the `create_overviews()` function to generate the required overview layers correctly.

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
from connectivity import overview_info, create_overviews, connectedness, beri
```

**1. Inspecting Overview Information**    
You can inspect the overview metadata of a TIFF file using the `overview_info()` function:

```python
overview_info(grids)
```
```
File: ../data/transgrids/1990.tif
Resolution: 588 x 516
Bands: 19
CRS: EPSG:4326

Overview Information:
  Band 1 overviews: [2, 4, 8, 16, 32]
  Overview resolutions:
    Level 2: 294 x 258
    Level 4: 147 x 129
    Level 8: 74 x 65
    Level 16: 37 x 33
    Level 32: 19 x 17
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
Use the `create_overviews()` function to generate and embed overviews into the TIFF file:

```python
create_overviews(
    input_raster = "raster_file.tif",
    output_raster = None, # keep None to update the file inplace
    overview_levels = [2, 4, 8, 16, 32]
)
```

**3. Calculating connectedness or BERI**   

The maximum distance the algorithm searches for cells (in the condition raster) to calculate connectivity is computed as:
`max_distance = outer_window * max(levels) * resolution`. For example, with a 1 km resolution raster, a max-level of 32, and an `outer_window` of 11, the resulting search distance is:

distance = outer_window × max_level × resolution
         = 11 × 32 × 1 km
         = 352 km  


*Connected Habitat (Connectedness)*
To compute connected-habitat (plain connectedness), you only need a habitat condition raster.

Use option argument to generate the connected condition from connectedness and input condition:    
1. connectedness    
2. connectedness * condition    
3. sqrt(connectedness * condition) (default)    

```python
connd = connectedness(
    condition_file = "./data/condition.tif",
    lambdas = [2, 20, 200],
    max_cost = 2.0, 
    window_size = 5, 
    outer_window = 11,
    levels = [2, 4, 8, 16, 32], 
    sigma = 1,
    option = 3,
    filename = "./results/connected_habitat.tif"
)
```

*PARC-Connectedness*    
To compute PARC-connectedness, provide both:
* a habitat condition raster, and
* a protected-areas proportion raster.    
     
When `pa_file` (proportion of protected-areas in each cell) is supplied, the function automatically returns PARC-connectedness instead of standard connectedness.

```python
parcc = connectedness(
    condition_file = "./data/condition.tif",
    pa_file = "./data/pa_proportion.tif",
    lambdas = [2, 20, 200],
    max_cost = 2.0, 
    window_size = 5, 
    outer_window = 11,
    levels = [2, 4, 8, 16, 32], 
    sigma = 1,
    filename = "./results/parc_connectedness.tif"
)
```

*BERI*    
To compute BERI, you must provide:
* a condition raster
* a current GDM transgrid
* one or more future transgrid scenarios

```python
beris = beri(
    condition_file = "./data/condition.tif",
    current_file = "./data/transgrids/1990.tif",
    future_files = ["./data/transgrids/IPS50_45.tif", "./data/transgrids/GFD50_85.tif"],
    lambdas = [2, 20, 200], 
    max_cost = 2.0, 
    window_size = 5, 
    outer_window = 11,
    levels = [2, 4, 8, 16, 32],
    sigma = 1,
    filename = "./results/berri.tif"
)
```
