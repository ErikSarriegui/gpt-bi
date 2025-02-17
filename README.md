```
git clone https://github.com/ErikSarriegui/gpt-bi

cd gpt-bi

pip install -r requirements.txt

torchrun --nproc_per_node=<n_gpus> train.py
```
