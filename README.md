## Description


## Prerequisites
FAReviews uses [python3](https://www.python.org) for argumentation mining and [prolog](https://www.swi-prolog.org) for argument reasoning. The required prolog scripts can be found in the folder "argue".

To install the required Python 3 packages use:

```bash
pip3 install -r requirements.txt
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
Creates: 
```
1_prods.pkl
1_reviews.csv
```

Create the matrix with distance metrics:
```bash
python3 graph_creation3.py 
```

Creates:

```
1_prods_mc.pkl
```

### Argument Reasoning

Download the ```argue``` folder, then run the following code:
```
cd argue
swipl server.pl
?- server(3333).
```

Solve the Argumentation Graph

```bash
python3 graph_creation_3.py 
```

Creates:
```
reviews_res.csv
results.pkl
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

Run the script for example as follows, after you have started the prolog server:
```
python3 FAReviews.py -f Data/mydata.csv -nc 4 -cs 50 -bs 40 -trt 0.10 -sn MyOutputFolder
```

The script will output all files described in run method 1, and will print which part it is currently working on and how long the finished parts have taken to complete. **Note that if you use the textrank threshold, tokens with a textrank score below the threshold are not used and therefore not saved in any of the output files.**
