:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(disco).

term_expansion(A->B, attacks(A, B)).

graph(1,                                        % 1 stable model
      [ b -> c,
        c -> b,
        d -> c,
        d -> b,
        c -> a,
        b -> a
      ]).
graph(2,                                        % 3 stable models
      [ a -> b,
        b -> a,
        b -> c,
        c -> d,
        d -> e,
        e -> c
      ]).
graph(Disco, Edges) :-
    disco_graph(Disco, Edges).

ddg(Graph) :-
    disco_graph(Graph, Edges),
    print_term(graph(Graph, Edges), [right_margin(1 000)]).

:- table ( node/2, nodes/2, attacks/3,
           state/4, model/2, label/3
         ).

model(Graph, Nodes) :-
    nodes(Graph, Nodes),
    label(Nodes, Graph, Nodes).

label([], _, _).
label(Unassigned, Graph, Nodes) :-
    select(N=_, Unassigned, Unassigned1),
    state(Graph, N, _, Nodes),
    include(unassigned, Unassigned1, Unassigned2),
    label(Unassigned2, Graph, Nodes).

unassigned(_=State) :-
    var(State).

state(Graph, X, Value, Nodes) :-
    member(X=Value0, Nodes),
    value(Value),
    (   Value == Value0
    ;   Value0 = Value,
        rule(Value, Graph, X, Nodes)
    ).

value(in).
value(out).
value(undec).

rule(in, Graph, X, Nodes) :-
    all_attackers_out(Graph, X, Nodes).
rule(out, Graph, X, Nodes) :-
    attacks(Graph, Y, X),
    state(Graph, Y, in, Nodes).
rule(undec, Graph, X, Nodes) :-
    not_exists(all_attackers_out(Graph, X, Nodes)),
    not_exists(an_attacker_in(Graph, X, Nodes)).

all_attackers_out(Graph, X, Nodes) :-
    all_attackers(Graph, X, Ys),
    all_out(Ys, Graph, Nodes).

all_attackers(Graph, X, Ys) :-
    findall(Y, attacks(Graph, Y, X), Ys).

all_out([], _, _).
all_out([H|T], Graph, Nodes) :-
    state(Graph, H, out, Nodes),
    all_out(T, Graph, Nodes).

an_attacker_in(Graph, X, Nodes) :-
    attacks(Graph, Y, X),
    state(Graph, Y, in, Nodes).

attacks(Graph, X, Y) :-
    graph(Graph, Edges),
    member((X->Y), Edges).

nodes(Graph, List) :-
    findall(X=_, node(Graph, X), List).

node(Graph, X) :-
    graph(Graph, Edges),
    member((A->B), Edges),
    (   X = A
    ;   X = B
    ).
