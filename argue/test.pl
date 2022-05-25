:- use_module(library(http/json)).
:- use_module(argue_json).

json_read_file(File, JSON) :-
    setup_call_cleanup(
        open(File, read, In),
        json_read_dict(In, JSON,
                       [ value_string_as(atom)
                       ]),
        close(In)).

test(File) :-
    json_read_file(File, Graph),
    argue_json(Graph, Model),
    pp(Model).
