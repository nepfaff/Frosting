set -e
set -o pipefail

# Create virtual environment
python -m venv .venv
. .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt231/download.html --upgrade

cd gaussian_splatting/submodules/diff-gaussian-rasterization/
pip install -e .

cd ../simple-knn/
pip install -e .

cd ../../../
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install -e .
cd ../
