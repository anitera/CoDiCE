# Setting up a Clean Environment

## Using Conda:
If you're using conda (from Anaconda or Miniconda distributions), you can set up a new environment as follows:
```
conda create -n demo_trustce python=3.10
```
Activate the environment:
```
conda activate demo_trustce
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
pyenv virtualenv 3.10.0 demo_trustce
```
Activate the environment:
```
pyenv activate demo_trustce
```

# Installing Your Framework
Once you have your clean environment set up:

Install the framework:
```
pip install trustce
```


## Contribution
We welcome contributions to `trustCE`! If you'd like to contribute, please fork the repository and submit a pull request. For major changes or feature requests, please open an issue first to discuss your ideas.

## License
`trustCE` is available under the MIT License. See the `LICENSE` file for more details.
