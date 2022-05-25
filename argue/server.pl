:- use_module(library(http/http_server)).
:- use_module(argue_json).

server(Port) :-
    http_server([ port(Port)
                ]).

:- http_handler(root(argue), argue, []).

argue(Request) :-
    http_read_json_dict(Request, Graph,
                        [ value_string_as(atom)
                        ]),
    argue_json_all(Graph, Model),
    reply_json_dict(Model).
