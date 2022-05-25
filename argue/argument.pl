:- use_module(library(aggregate)).

term_expansion(A->B, attacks(A, B)).

b -> c.
c -> b.
d -> c.
d -> b.
c -> a.
b -> a.

:- table in/1, out/1, undec/1, all_attackers_out/1, an_attacker_in/1, node/1.

in(X) :-
    node(X),
    all_attackers_out(X).
out(X) :-
    node(X),
    attacks(Y,X),
    in(Y).
undec(X) :-
    node(X),
    tnot(all_attackers_out(X)),
    tnot(an_attacker_in(X)).

all_attackers_out(X) :-
    foreach(attacks(Y,X),
            out(Y)).
an_attacker_in(X) :-
    attacks(Y, X),
    in(Y).

node(X) :-
    attacks(X, _).
node(X) :-
    attacks(_, X).
