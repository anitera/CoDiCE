# Setting up a Clean Environment

## Using Conda:
If you're using conda (from Anaconda or Miniconda distributions), you can set up a new environment as follows:
```
conda create -n demo_codice python=3.10
```
Activate the environment:
```
conda activate demo_codice
```
Or
## Using pyenv
If you prefer `pyenv`:

Install the desired Python version:
```
pyenv install 3.10.0
```
Create a new virtual environment:
```
pyenv virtualenv 3.10.0 demo_codice
```
Activate the environment:
```
pyenv activate demo_codice
```

# Installing Your Framework
Once you have your clean environment set up:

Install the framework:
```
pip install codice
```

# Running example notebooks
If you want to use local jupyter notebook run the following commands:
```
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=demo_codice
pip install jupyter
```
After that type in the root folder:
```
jupyter notebook
```


## Contribution
We welcome contributions to `codice`! If you'd like to contribute, please fork the repository and submit a pull request. For major changes or feature requests, please open an issue first to discuss your ideas.

## License
`codice` is available under the MIT License. See the `LICENSE` file for more details.
