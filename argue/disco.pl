:- module(disco,
          [ disco_graph/2                       % +Name, -Edges
          ]).
:- use_module(library(http/json)).

file_json(File, JSON) :-
    setup_call_cleanup(
        open(File, read, In),
        json_read_dict(In, JSON),
        close(In)).

%:- table disco_graph/2.

disco_graph(Name, Edges) :-
    atom(Name),
    format(atom(File), 'disco/~w.json', Name),
    exists_file(File),
    file_json(File, JSON),
    maplist(edge, JSON.edges, Edges).
disco_graph(Name, Edges) :-
    var(Name),
    expand_file_name('disco/*.json', Files),
    member(File, Files),
    file_base_name(File, Base),
    file_name_extension(Name, json, Base),
    disco_graph(Name, Edges).

edge(JSON, (From->To)) :-
    atom_string(From,JSON.data.source),
    atom_string(To,JSON.data.target).
