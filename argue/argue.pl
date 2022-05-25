:- module(argue,
          [ edges_model/2               % +Edges, -Model
          ]).
:- use_module(library(lists)).
:- use_module(library(apply)).

:- thread_local graph/2.

edges_model(Edges, Model) :-
    setup_call_cleanup(
        asserta(graph(g, Edges), Ref),
        model(g, Model),
        (   erase(Ref),
            abolish_all_tables
        )).

:- table ( node/2, nodes/2, attacks/3,
           state/4, model/2, label/3
         ).

%!  model(+GraphID, -Model) is nondet.
%
%   True when Model is a valid model (a list of NodeID=State) for the graph
%   with id GraphID.  The graph itself is represented as a fact of the format
%
%       graph(GraphID, ListOfEdges)
%
%   where each Edge is a term `Node1 -> Node2`.  The notebook argue.swinb
%   provides examples from the slides and _disco_ reasoner.

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

%!  state(+GraphID, +NodeID, ?State, ?Model)
%
%   This is the central predicate of our reasoner.  If NodeID has no assigned
%   state it tries to assign a state and propagate the consequences through
%   the graph.  If NodeID has State on entry we simply succeed.  Note that
%   we should __not__ call the rules in this case as this will eventually
%   cause the rules of other nodes to call this one again, leading to a
%   loop similar to `p <- q, q <- r, r <-p`.  Using tabled Prolog this
%   does not loop but, logically correctly, fails.

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

%!  rule(+State, +GraphID, +NodeID, ?Nodes)
%
%   Verify that we can set the state of NodeID to State.  This implemets the
%   rules described in the slide __Argument Labellings__.  It should be totally
%   obvious that this is equivalent to the slide.  Note that the execution is
%   rather complex.  not_exists/1 implements sound negation for non-ground goals
%   which we need because Nodes may not be ground.  The rules are also recursive
%   through the negations.  This requires tabling with well founded semantics
%   as developed by the XSB team and brought to SWI-Prolog with the help of
%   Teri Swift and David S. Warren, supported by Benjamin Grosof and Kyndi.

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

% The stuff below provides the really basic access to the graph properties
% we are interested in: the nodes in the graph to create the model and
% the _attack_ relation between two nodes.

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

/** <examples>
?- edges_model([a->b,b->a],Model).
*/
