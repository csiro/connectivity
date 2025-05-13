## multires-connectivity: a multi-resolution landscape connectivity algorithm

This algorithm operates on the overview layers of Cloud Optimized GeoTIFF (COG) files. Please ensure that these overview layers are generated using `mean` aggregation, not `nearest neighbor` resampling.

### Installation

You need to load the module and create an environment if you don't alreay have one.

```bash
module load Python/3.12.3
module load miniconda3/23.3.1
module load rust/1.84.1
```

```bash
conda create --name spatial # or any env name
conda init bash
```
Installing the library into the environent:

```bash
source activate spatial

cd ~/multires_connectivity
```

Option 1: Install in editable mode (develop locally)

```bash
maturin develop
```

Option 2: Build a wheel and install it

```bash
maturin build --release
```
Then find the wheel under `target/wheels/`, and install it with:

```bash
pip install target/wheels/multires_connectivity-*.whl
```

### Example

```python
from multires_connectivity import connectivity, create_overviews, overview_info
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
Use the create_overviews() function to generate and embed overviews into the TIFF file.

```python
create_overviews(
    input_raster = "file.tif",
    output_raster = None, # keep None to update layer inplace
    overview_levels = [1, 2, 4, 8, 16] # the overview levels
)
```

**3. Calculating connectivity or BERI**

```python
connd = connectivity(
    path = "cog_file.tif",
    disp_rate = 0.2,
    scale = 2.5, 
    nb_size = 3, 
    last_nb_size = 5,
    levels = [1, 2, 4, 8, 16],
    mask = True, 
    smooth = True,
    filename = ""
)

```