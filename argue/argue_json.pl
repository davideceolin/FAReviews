:- module(argue_json,
          [ argue_json/2,               % +Graph, -Model
            argue_json_all/2            % +Graph, -Models
          ]).
:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(argue).

/** <module> JSON frontend for argumentation reasoner

Example input:

```
{'directed': True,
 'multigraph': False,
 'graph': {},
 'nodes': [{'weight': 1, 'id': 0}, {'id': 1}, {'id': 2}],
 'links': [{'weight': 3.0, 'source': 0, 'target': 1},
           {'weight': 7.5, 'source': 1, 'target': 2}
          ]}
```
*/

%!  argue_json_all(+Graph, -Models)

argue_json_all(Graph,
               json{time:Time,
                    models:Models}) :-
    statistics(cputime, T0),
    findall(Model, argue_json(Graph, Model), Models0),
    sort(weight, >=, Models0, Models),
    statistics(cputime, T1),
    Time is T1-T0.

%!  argue_json(+Graph, -Model)
%

argue_json(Graph, JModel) :-
    json_edges(Graph, Edges),
    edges_model(Edges, Model),
    model_json(Model, Graph, JModel).

json_edges(Graph, Edges) :-
    findall((Source->Target),
            graph_edge(Graph, Source, Target, _Weight),
            Edges).

graph_edge(Graph, Source, Target, Weight) :-
    member(Link, Graph.links),
    _{weight:Weight, source:Source, target:Target} :< Link.

%!  model_json(+Model, +Graph, -GraphEx)
%
%   Model is a list of [NodeID = State]

model_json(Model, Graph, GraphEx) :-
    dict_create(Dict, #, Model),
    add_states(Graph.nodes, Dict, NewNodes, In, Out, UnDec),
    length(Graph.nodes, NodeCount),
    NodeAvg is In/NodeCount,
    link_weights(Graph, Dict, LinkWeights),
    GraphEx = Graph.put(_{ nodes:NewNodes,
                           node_weights: #{in:In, out:Out, undec:UnDec},
                           link_weights: LinkWeights,
                           weight:NodeAvg
                         }).

add_states(Nodes, Model, NodesEx, In, Out, Undec) :-
    add_states(Nodes, Model, NodesEx, 0, In, 0, Out, 0, Undec).

add_states([], _, [], In, In, Out, Out, Undec, Undec).
add_states([H0|T0], Model, [H|T], In0, In, Out0, Out, Undec0, Undec) :-
    Id = H0.id,
    State = Model.get(Id),
    H = H0.put(state, State),
    (   W = H0.get(weight)
    ->  true
    ;   W = 1
    ),
    add_weight(State, W, In0, In1, Out0, Out1, Undec0, Undec1),
    add_states(T0, Model, T, In1, In, Out1, Out, Undec1, Undec).

add_weight(in,    W, In0, In, Out, Out, Undec, Undec) :- In is In0+W.
add_weight(out,   W, In, In, Out0, Out, Undec, Undec) :- Out is Out0+W.
add_weight(undec, W, In, In, Out, Out, Undec0, Undec) :- Undec is Undec0+W.

link_weights(Graph, Model, Weights) :-
    maplist(link_weight(Model), Graph.links, ByLink),
    keysort(ByLink, Sorted),
    group_pairs_by_key(Sorted, Grouped),
    maplist(sum_values, Grouped, Weights).

link_weight(Model, Link, #{source:SourceState, target:TargetState}-Weight) :-
    _{weight:Weight, source:Source, target:Target} :< Link,
    SourceState = Model.get(Source),
    TargetState = Model.get(Target).

sum_values(Type-Weights, Dict) :-
    sum_list(Weights, Weight),
    Dict = Type.put(sum, Weight).
