## multires-connectivity: A multi-resolution landscape connectivity algorithm

This algorithm operates on the overview layers of Cloud Optimized GeoTIFF (COG) files. Please ensure that these overview layers are generated using mean aggregation, not nearest neighbor resampling.

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
```

Option 1: Install in editable mode (develop locally)

```bash
cd ~/multires_connectivity
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

