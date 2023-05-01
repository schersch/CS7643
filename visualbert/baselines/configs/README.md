Running the Baseline Models

Steps to run baseline models:
1. Downlod data zip from [here](https://gtvault-my.sharepoint.com/personal/ghaglund3_gatech_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fghaglund3%5Fgatech%5Fedu%2FDocuments%2Fhm%5Fdata%2Ezip&parent=%2Fpersonal%2Fghaglund3%5Fgatech%5Fedu%2FDocuments&ga=1)
2. Install MMF following instructions [here](https://mmf.sh/docs/)
3. Follow instructions specific to running MMF on different models specific to the hateful memes problem set [here](https://github.com/facebookresearch/mmf/tree/main/projects/hateful_memes)
4. Models were not included due their large size, however the results can be replicated by rerunning MMF using the provided config files in configs/

Troubleshooting Guidance
1. MMF is a very OS and environment specific library. If you see issues in installation ensure that you are running in a python 3.7 environment using dated python dependencies that are required of the library such as pytorch. 

While Windows is supported, there are a number of incompatibilities with the latest OS. Google collab is also not recommended based on analysis due to the lack of older dependency support and the difficulties of running with virtual package managers such as conda/venv. Would recommend running directly on GCP.

2. If issues are hit while converting the zip file, it might be due to how the zip must be specified. Ensure that the zipped file is in the format of `data/img, data/{json files}
