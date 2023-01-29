from dataclasses import dataclass
from fractions import Fraction
from typing import Union, Mapping


@dataclass
class NumLiteral:
    value: Fraction

    def __init__(self, *args):
        self.value = Fraction(*args)


@dataclass
class BinOp:
    operator: str
    left: "AST"
    right: "AST"


@dataclass
class Boolean:
    value: bool


@dataclass
class Variable:
    name: str
    #defining slicing method for string slicing
    def slicing(self,name,start_index:NumLiteral,end_index:NumLiteral):
        
        if start_index>len(name)-1:
            raise IndexError
        if end_index<=start_index:
            raise IndexError
        if end_index>len(name):
            raise IndexError
        else:
            string_slice=name[start_index:end_index]
            return string_slice



@dataclass
class Let:
    var: "AST"
    e1: "AST"
    e2: "AST"


AST = NumLiteral | BinOp | Variable | Let

Value = Fraction


class InvalidProgram(Exception):
    pass

#Modified Binop("+",left,right) for typechecking of strings for concatenation
def typeof(s: AST):
    match s:
        case Variable(name):
            return "type string"
        case NumLiteral(value):
            return "type Fraction"
        case Boolean(value):
            return "type boolean"
        case BinOp("+", left, right): 
            if typeof(left) == "type boolean" or typeof(right) == "type boolean":
                raise TypeError()
            elif typeof(left)!=typeof(right):
                raise TypeError()
            else:
                if typeof(left) == "type string" and typeof(right) == "type string":
                    return "type string"
                else:
                    return TypeError()


def eval(program: AST, environment: Mapping[str, Value] = None) -> Value:
    typeof(program)
    if environment is None:
        environment = {}
    match program:
        case NumLiteral(value):
            return value
        case Variable(name):
            return name
        case Boolean(value):
            return value
        case Let(Variable(name), e1, e2):
            v1 = eval(e1, environment)
            return eval(e2, environment | {name: v1})
        case BinOp("+", left, right):
            return eval(left, environment) + eval(right, environment)
        case BinOp("-", left, right):
            return eval(left, environment) - eval(right, environment)
        case BinOp("*", left, right):
            return eval(left, environment) * eval(right, environment)
        case BinOp("/", left, right):
            return eval(left, environment) / eval(right, environment)
        case BinOp(">", left, right):
            return eval(left, environment) > eval(right, environment)
        case BinOp("<", left, right):
            return eval(left, environment) < eval(right, environment)
        case BinOp("==", left, right):
            return eval(left, environment) == eval(right, environment)
            
    raise InvalidProgram()


def test_eval():
    e1 = NumLiteral(2)
    e2 = NumLiteral(7)
    e3 = NumLiteral(9)
    e4 = NumLiteral(5)
    e5 = BinOp("+", e2, e3)
    e6 = BinOp("/", e5, e4)
    e7 = BinOp("*", e1, e6)
    assert eval(e7) == Fraction(32, 5)


def test_let_eval():
    a = Variable("a")
    e1 = NumLiteral(5)
    e2 = BinOp("+", a, a)
    e = Let(a, e1, e2)
    assert eval(e) == 10
    e = Let(a, e1, Let(a, e2, e2))
    assert eval(e) == 20
    e = Let(a, e1, BinOp("+", a, Let(a, e2, e2)))
    assert eval(e) == 25
    e = Let(a, e1, BinOp("+", Let(a, e2, e2), a))
    assert eval(e) == 25
    e3 = NumLiteral(6)
    e = BinOp("+", Let(a, e1, e2), Let(a, e3, e2))
    assert eval(e) == 22
    
#Sanity Check for concatenation
def test_concat():
    a=Variable("hello")
    b=Variable("world")
    c=BinOp("+",a,b)
    assert eval(c)== "helloworld"
    
#Sanity Check for slicing
def test_slice(a:Variable):
    
    Variable.slicing(Variable,a,1,4)

test_concat()
test_slice("hello")
