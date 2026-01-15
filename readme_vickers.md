# this is what Vickers did..


conda create -n connectivity-py312 python=3.12 pip -c conda-forge -y
# which lands in: /scratch3/vic039/conda-envs/connectivity-py312
conda activate connectivity-py312

pip install -U pip wheel maturin

cd ~/repos/connectivity

module load rust/1.84.1

<!-- maturin develop --release  
# do this as develop rather than build, and don't need to build a wheel and install it

# then test everything works:
python -c "import connectivity; print('connectivity import OK')"
pip show connectivity | sed -n '1,20p'
python -c "import connectivity, inspect, os; print('loaded from:', inspect.getfile(connectivity))"
# that all doesn't work, and need to build the wheel and install from there: -->

maturin build --release
pip install --force-reinstall target/wheels/connectivity-*.whl

