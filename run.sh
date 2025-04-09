./prebuild.sh
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONPATH=. 
python app/main.py --config=./deep.yaml
