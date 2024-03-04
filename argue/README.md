# Theory

   - https://users.cs.cf.ac.uk/CaminadaM/publications/IJCAI19_presentation.pdf

# Dataset

  - http://corpora.aifdb.org/rrd
  - http://disco.cs.cf.ac.uk/examples/

# Running the HTTP service

  - Get a recent version of SWI-Prolog 8.1.x (currently 8.1.27)
  - run this to start the server on port 3333, only accessible from
    `localhost`.  If you want to access from another machine, drop
    `localhost:`.

        swipl server.pl
	?- server(localhost:3333).
  <!-- markdown-link-check-disable -->
  - To get the results, issue a POST request with the graph as JSON
    to http://localhost:3333/argue.  Using curl this is for example
    (one line)

        curl -X POST -H "Content-Type: application/json" \
            -d @g01.json http://localhost:3333/argue    <!-- markdown-link-check-disable-line -->

The server can solve multiple models concurrently. Just make sure to run
HTTP requests concurrently.
