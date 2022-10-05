## Description


## Prerequisites
FAReviews uses [python3](https://www.python.org) for argumentation mining and [prolog](https://www.swi-prolog.org) for argument reasoning. The required prolog scripts can be found in the folder "argue".

To install the required Python 3 packages use:

```bash
pip3 install -r requirements.txt
pip3 install --upgrade spacy
pip3 install pytextrank
```


Download data (todo: download nltk stopwords)

```bash
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
wget -c "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/AMAZON_FASHION_5.json.gz"
python -m spacy download en_core_web_md
```

## Run method 1:
### Argument Mining
Perform feature extraction:
```bash
python3 compute_scores2.py 
```
The script will ask you to provide the number of jobs, chunks, batch size, textrank threshold, and folder for output. It creates in the output folder: 
```
[datafile name]_prods.pkl
[datafile name]_reviews.csv
```

Create the matrix with distance metrics:
```bash
python3 graph_creation3.py 
```
The script will ask you to provide the csv file with review data ([datafile name]_reviews.csv), the pkl file with the product list ([datafile name]_prods.pkl), the number of cores to use, and folder for output. It creates in the output folder:

```
[datafile name]_prods_mc.pkl
```

### Argument Reasoning

Download the ```argue``` folder, then run the following code to start the server:
```
cd argue
swipl server.pl
?- server(3333).
```

While the server is running, solve the Argumentation Graph. 
```bash
python3 graph_creation_3.py 
```
The script will ask you to provide the csv file with review data ([datafile name]_reviews.csv), the pkl file with the product list and matrices and clusters ([datafile name]_prods_mc.pkl), the number of cores to use, the folder to use for the output, and whether or not to save the figures of the created graphs to png. It creates in the output folder:
```
[datafile name]_reviews_results.csv
[product asin].png
[product asin_labels].png
```

## Run method 2:
The three scripts described above can be ran sequentially using the FAReviews.py. This allows the user to provide the input data and several input parameters. In order for the argument reasoning part to be able to start from FAReviews.py, you have to make sure to have the prolog server running. <br />
The following arguments can be provided to FAReviews.py (only `-f` is a required argument, the others are optional):

- `-f`: Provide the location of the input data file (csv expected). Required argument.
- `-nc`: Number of cores to use for the various processes that are ran in parallel. Default = 8.
- `-cs`: Chunk size used in compute scores. Default = 100.
- `-bs`: Batch size used in compute scores. Default = 20.
- `-trt`: Minimum textrank score threshold for the tokens to be used. Tokens with a textrank score below the threshold are not used. The threshold is used in compute scores, and the resulting output is passed to the scripts that follow. Default = 0.0.
- `-sn`: Name of the output folder (within the current folder) where you want to save the output. If it does not yet exist, it will be created. Default is *Output*.
- `-si`: True/False If true, also save the output of compute scores and run_graph. If false, only the output of graph_creation_3 is saved. Default is False.
- `-sf`: True/False. Option to save the constructed graphs to png per product. Default is False.

Run the script for example as follows, after you have started the prolog server:
```
python3 FAReviews.py -f Data/mydata.csv -nc 4 -cs 50 -bs 40 -trt 0.10 -sn MyOutputFolder
```

The script will output the final results of graph_creation_3 (and intermediate results/graphs if respective arguments are set), and will print which part it is currently working on and how long the finished parts have taken to complete. **Note that if you use the textrank threshold, tokens with a textrank score below the threshold are not used and therefore not saved in any of the output files.**
