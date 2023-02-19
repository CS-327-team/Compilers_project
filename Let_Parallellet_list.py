from dataclasses import dataclass
from fractions import Fraction
from typing import Union, Mapping, List

@dataclass
class Cons:
    head: 'AST'
    tail: 'AST'

@dataclass
class IsEmpty:
    lst: 'AST'

@dataclass
class Head:
    lst: 'AST'

@dataclass
class Tail:
    lst: 'AST'

@dataclass
class NumLiteral:
    value: Fraction
    def __init__(self, *args):
        self.value = Fraction(*args)

@dataclass
class BinOp:
    operator: str
    left: 'AST'
    right: 'AST'

@dataclass
class Variable:
    name: str
#implementing let
@dataclass
class Let:
    var: 'AST'
    e1: 'AST'
    e2: 'AST'

#implementing parallel let
#Parallel let allows the values of two or more variables to be as simultaneously, in parallel, rather than sequentially
@dataclass
class ParallelLet:
    var: 'AST'
    e1: 'AST'
    e2: 'AST'

AST = NumLiteral | BinOp | Variable | Let | ParallelLet
Value = Fraction

class InvalidProgram(Exception):
    pass

def eval(program: AST, environment: Mapping[str, Value] = None) -> Value:
    if environment is None:
        environment = {}
    match program:
        case NumLiteral(value):
            return value # Returns the value of the NumLiteral
        case Variable(name):
            if name in environment:
                return environment[name] # Returns the value of the variable if it exists in the environment
            raise InvalidProgram() # Raises an error if the variable is not in the environment
        case BinOp("+", left, right):
            return eval(left, environment) + eval(right, environment) 
        case BinOp("-", left, right):
            return eval(left, environment) - eval(right, environment) 
        case BinOp("*", left, right):
            return eval(left, environment) * eval(right, environment) 
        case BinOp("/", left, right):
            return eval(left, environment) / eval(right, environment) 
        #adding case for let
        case Let(Variable(name), e1, e2):
            v1 = eval(e1, environment) # Evaluates the first expression
            return eval(e2, environment | { name: v1 }) # Evaluates the second expression with the first expression's result bound to the variable
        #adding case for parallel let
        case ParallelLet(Variable(name), e1, e2):
            v1 = eval(e1, environment) # Evaluates the first expression
            # Evaluates the second expression with the first expression's result bound to the variable in parallel
            return eval(e2, environment | { name: v1 }) 
        case Cons(head, tail):
            return [eval(head, environment)] + eval(tail, environment)
        case IsEmpty(lst):
            return len(eval(lst, environment)) == 0
        case Head(lst):
            return eval(lst, environment)[0]
        case Tail(lst):
            return eval(lst, environment)[1:]
      
    raise InvalidProgram() # Raises an error if the program is invalid

def test_eval():
    e1 = NumLiteral(2)
    e2 = NumLiteral(7)
    e3 = NumLiteral(9)
    e4 = NumLiteral(5)
    e5 = BinOp("+", e2, e3)
    e6 = BinOp("/", e5, e4)
    e7 = BinOp
    
def test_let_eval():
    a  = Variable("a")
    e1 = NumLiteral(5)
    e2 = BinOp("+", a, a)
    e  = Let(a, e1, e2)
    assert eval(e) == 10
    e  = Let(a, e1, Let(a, e2, e2))
    assert eval(e) == 20
    e  = Let(a, e1, BinOp("+", a, Let(a, e2, e2)))
    assert eval(e) == 25
    e  = Let(a, e1, BinOp("+", Let(a, e2, e2), a))
    assert eval(e) == 25
    e3 = NumLiteral(6)
    e  = BinOp("+", Let(a, e1, e2), Let(a, e3, e2))
    assert eval(e) == 22

def test_parallel_let_eval():
    a  = Variable("a")
    b  = Variable("b")
    e1 = NumLiteral(5)
    e2 = BinOp("+", a, b)
    e3 = BinOp("*", e1, e1)
    e  = ParallelLet(a, e1, b, e3, e2)
    assert eval(e) == 30
    e  = ParallelLet(a, e1, b, e2, e2)
    assert eval(e) == 20
    e  = ParallelLet(a, e2, b, e2, BinOp("+", a, b))
    assert eval(e) == 40
    e  = ParallelLet(a, e2, b, BinOp("+", a, b), e2)
    assert eval(e) == 40

#cons: this takes an argument and returns a new list whose head is the argument and whose tail is the list it is called on. This means that it adds a new element to the beginning of the list.
#is_empty: this returns a boolean indicating whether the list it is called on is empty or not.
#head: this returns the first element of the list it is called on.
#tail: this returns a new list consisting of all but the first element of the list it is called on. In other words, it returns the remaining elements of the list after the first one has been removed.


def test_list_list():
    e1 = NumLiteral(2)
    e2 = NumLiteral(7)
    e3 = NumLiteral(9)
    e4 = Cons(e1, Cons(e2, Cons(e3, NumLiteral(0))))
    e5 = IsEmpty(e4)
    e6 = Head(e4)
    e7 = Tail(e4)
    assert eval(e5) == False
    assert eval(e6) == 2
    assert eval(e7) == [7, 9, 0]
    e8 = Cons(NumLiteral(5), Cons(BinOp("*", e1, e2), e7))
    assert eval(e8) == [5, 14, 7, 9, 0]

