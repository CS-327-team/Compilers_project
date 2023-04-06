from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, List as List

digit_list = "1234567890"
alphabet_list = "ABCDEFGHIJKLOMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
variable_list = []

@dataclass
class EndOfStream(Exception):
    pass

@dataclass
class Num:
    n: int


@dataclass
class Bool:
    b: bool


@dataclass
class Keyword:
    word: str


@dataclass
class Identifier:
    word: str


@dataclass
class Operator:
    operator: str



@dataclass
class Paranthesis:
    paran: str


@dataclass
class Delimiter:
    delim: str


@dataclass
class String:
    string: str

Token = Num | Bool | Keyword | Identifier | Operator | Paranthesis | Delimiter|String


@dataclass
class EndOfTokens(Exception):
    pass


def convert_to_token(c):
    if c == "True" or c == "False":
        return Bool(c)
    elif c in ["=", ">", "<", "+", "-", "*", "/", "!=", "<=", ">=", "^", "%"]:
        return Operator(c)
    elif c in "if then else end while do done for".split():
        return Keyword(c)
    return Identifier(c)


@dataclass
class EndOfToken(Exception):
    pass


@dataclass
class TokenError(Exception):
    pass


operations = ["=", ">", "<", "+", "-", "*", "/", "!=", "<=", ">=", "%", "^"]
keywords = "if then else while print for from to def let in".split()
delimiters = ['"', ";"]

class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = -1
        self.current_char = None
        self.advance()

    def advance(self):
        self.pos += 1
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None

    def tokenize(self):
        tokens = []
        while self.current_char != None:
            match self.current_char:
                case op if op in ["+", "-", "*", "/", "%", "^"]:
                    tokens.append(Operator(op))
                    self.advance()
                case op if op in ["!", "=", ">", "<"]:
                    self.advance()
                    if self.current_char == "=":
                        op += self.current_char
                        self.advance()
                    tokens.append(Operator(op))
                case space if space in " \t\n":
                    self.advance()
                case num if num in digit_list:
                    num_temp = ""
                    while self.current_char != None and self.current_char in digit_list:
                        num_temp += self.current_char
                        self.advance()
                    tokens.append(Num(int(num_temp)))
                case s if s in alphabet_list:
                    temp_str = ""
                    while (
                        self.current_char != None
                        and self.current_char in alphabet_list + digit_list
                    ):
                        temp_str += self.current_char
                        self.advance()
                    tokens.append(
                        Keyword(temp_str)
                    ) if temp_str in keywords else tokens.append(
                        Bool(True if temp_str == "True" else False)
                    ) if temp_str in "True False".split() else tokens.append(
                        Identifier(temp_str)
                    )
                case "{":
                    temp = ""
                    num = 1
                    self.advance()
                    while num > 0:
                        if self.current_char == "{":
                            num += 1
                        if self.current_char == "}":
                            num -= 1
                        if type(self.current_char) == str:
                            temp += self.current_char
                        self.advance()
                    tokens.append(Lexer(temp[:-1]).tokenize())
                case s if s in "()":
                    tokens.append(Paranthesis(s))
                    self.advance()

                case "\'":
                    temp = ""
                    self.advance()
                    while self.current_char != "\'":
                        temp += self.current_char
                        self.advance()
                    self.advance()
                    tokens.append(String(temp))

                case s if s in delimiters:
                    tokens.append(Delimiter(s))
                    self.advance()
                case _:
                    raise TypeError()
        return tokens


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
class Variable:
    name: str
    
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
class Var:
    name: str
    value: str | int | bool = None

@dataclass
class ForLoop:
    var: "AST"
    start: "AST"
    end: "AST"
    body: "AST"

@dataclass
class WhileLoop:
    cond: bool
    task: List

@dataclass
class Cons:
    head:'AST'
    tail:'AST'

@dataclass
class Isempty:
    lst:'AST'

@dataclass
class Head:
    lst:'AST'

@dataclass
class Tail:
    lst:'AST'


# Implementing If-Else statement
@dataclass
class If:
  cond : 'AST'
  true_branch : 'AST'
  false_branch : 'AST'

# Implementing the Boolean Type
@dataclass
class BoolLiteral:
    value : bool
    def __init__(self, value:bool):
        self.value = value

@dataclass
class Let:
    var: "AST"
    e1: "AST"
    e2: "AST"

# implementing the print function
@dataclass
class Print:
    exp: "AST"

@dataclass
class FnObject:
    params: List["AST"]
    body: "AST"

Value = Fraction | FnObject | bool | ForLoop | Let 

# Implementing functions
@dataclass
class FunCall:
    parameters: List[str]
    body: List['AST']

    def call(self, arguments: List[Value]) -> Value:
        if not self.body:      # to handle the case where body of the function is empty
            return None
        
        # the parameters passed in the function while calling it should be equal to the number arguments while defining it. 
        # this if statement checks this condition
        if len(arguments) != len(self.parameters):
            raise InvalidProgram()
        
        environment = dict(zip(self.parameters, arguments))
        result = None

        for expression in self.body:
            result = eval(expression, environment)  # here, expression can also be a function, thus calling recurssion

        return result 

@dataclass
class Put:
    var: "AST"
    e1: "AST"


@dataclass
class Get:
    var: "AST"

@dataclass
class ParallelLet:
    vars: List[Variable]
    exprs: List['AST']
    body: 'AST'

@dataclass
class Index:
    array: 'AST'
    index: 'AST'

@dataclass
class Append:
    array: 'AST'
    value: 'AST'

@dataclass
class Pop:
    array: 'AST'

@dataclass
class Concat:
    left: 'AST'
    right: 'AST'

@dataclass
class Assign:
    array: 'AST'
    index: 'AST'
    value: 'AST'

@dataclass
class MutableArray:
    elements: List['AST']



@dataclass
class Put:
    var: "AST"
    e1: "AST"


@dataclass
class Get:
    var: "AST"

@dataclass
class Seq:
    things: List["AST"]

AST = (
      NumLiteral
    | BinOp
    | Variable
    | Let
    | Put
    | Get
    | Seq
    | ParallelLet
    | FunCall
    | ForLoop
    | Index | Append | Pop | Concat | Assign | MutableArray
    | WhileLoop
    | Cons 
    | Isempty 
    | Head 
    | Tail

)

class InvalidProgram(Exception):
    pass


def typeof(s: AST):
    match s:
        case Variable(name):
            return "type string"
        case NumLiteral(value):
            return "type Fraction"
        case BoolLiteral(value):
            return "type boolean"
        case Var(name, value):
            flag = 1
            for var in variable_list:
                if var.name == name:
                    flag = 0
                    return (
                        "type boolean" * (type(var.value) == bool)
                        + "type string" * (type(var.value) == str)
                        + "type Fraction" * (type(var.value) == Fraction)
                    )
            if flag:
                return InvalidProgram("reference before assignment")
        case BinOp("+", left, right):
            if typeof(left) != typeof(right):
                return TypeError()
            elif typeof(left) == "type boolean":
                return TypeError()
            else:
                return typeof(left)
        case BinOp("-", left, right):
            if typeof(left) != "type Fraction" or typeof(right) != "type Fraction":
                raise TypeError()
            else:
                return "type Fraction"
        case BinOp("%", left, right):
            if typeof(left) == "type Fraction" and typeof(right) == "type Fraction":
                return "type Fraction"
            else:
                raise TypeError()
        case BinOp("^", left, right):
            if typeof(left) != "type Fraction" or typeof(right) != "type Fraction":
                raise TypeError()
            else:
                return "type Fraction"
        case BinOp("*", left, right):
            if typeof(left) != "type Fraction" or typeof(right) != "type Fraction":
                raise TypeError()
            else:
                return "type Fraction"
        case BinOp("/", left, right):
            if typeof(left) != "type Fraction" or typeof(right) != "type Fraction":
                raise TypeError()
            else:
                return "type Fraction"
        case BinOp(">", left, right):
            if typeof(left) == "type Fraction" and typeof(right) == "type Fraction":
                return "type boolean"
            else:
                raise TypeError()
        case BinOp("<", left, right):
            if typeof(left) == "type Fraction" and typeof(right) == "type Fraction":
                return "type boolean"
            else:
                raise TypeError()
        case BinOp("==", left, right):
            if typeof(left) == typeof(right):
                return "type boolean"
            else:
                raise TypeError()
        case BinOp(">=", left, right):
            if typeof(left) == "type Fraction" and typeof(right) == "type Fraction":
                return "type boolean"
            else:
                raise TypeError()
        case BinOp("<=", left, right):
            if typeof(left) == "type Fraction" and typeof(right) == "type Fraction":
                return "type boolean"
            else:
                raise TypeError()
        case BinOp("!=", left, right):
            if typeof(left) == "type Fraction" and typeof(right) == "type Fraction":
                return "type boolean"
            else:
                raise TypeError()
        case BinOp("=", left, right):
            flag = 1
            for var in variable_list:
                if var.name == left.name:
                    flag = 0
                    if type(var.value) == type(eval(right)):
                        return
            if flag:
                return
            raise TypeError("variables cannot change type")

        case If(cond, true_branch, false_branch):
            if typeof(cond) != "type boolean":
                raise TypeError("Invalid condition")
            return
        case Print(exp):
            return
    raise TypeError()

class Environment:
    envs: List

    def __init__(self):
        self.envs = [{}]

    def enter_scope(self):
        self.envs.append({})

    def exit_scope(self):
        assert self.envs
        self.envs=self.envs[:-1]

    def add(self, name, value):
        assert name not in self.envs[-1]
        self.envs[-1][name] = value

    def get(self, name):
        for env in reversed(self.envs):
            if name in env:
                return env[name]
        raise KeyError("reference before assignment")

    def update(self, name, value):
        for env in reversed(self.envs):
            if name in env:
                env[name] = value
                return env[name]
        self.add(name, value)
       

def eval(program: AST, environment: Environment) -> Value:
    # typeof(program)
    if environment is None:
        environment = Environment()
    match program:
        case NumLiteral(value):
            return value
        case Variable(name):
            return name
        case BoolLiteral(value):
            return value
        case Var(name, value):
            return environment.get(name)
        case Let(Var(name,value),left,right):
            environment.envs.append({name:eval(left,environment)})
            ans=eval(right,environment)
            environment.exit_scope()
            return ans
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
        case BinOp("%", left, right):
            return eval(left,environment) % eval(right,environment)
        case If(cond, true_branch, false_branch):
            if eval(cond, environment):
                for task in true_branch:
                    eval(task,environment)
            else:
                for task in false_branch:
                    eval(task,environment)
        case BinOp("^", left, right):
            return eval(left) ** eval(right)
        case BinOp("!=", left, right):
            return eval(left, environment) != eval(right, environment)
        case BinOp("<=", left, right):
            return eval(left, environment) <= eval(right, environment)
        case BinOp(">=", left, right):
            return eval(left, environment) >= eval(right, environment)
        case BinOp("=", left, right):
            right_eval = eval(right, environment)
            match left:
                case Var(name, value):
                    environment.update(name, right_eval)
                case _:
                    raise InvalidProgram()
        case ForLoop(var, start, end, body):
            start_val = eval(start, environment)
            end_val = eval(end, environment)
            environment.enter_scope()
            for j in range(int(start_val), int(end_val) + 1):
                eval(BinOp("=", var, NumLiteral(Fraction(j))),environment)
                for ast in body:
                    eval(ast, environment)
            environment.exit_scope()
        case WhileLoop(cond, task):
            environment.enter_scope()
            while eval(cond, environment) == True:
                for tas in task:
                    eval(tas, environment)
            environment.exit_scope()
        case Cons(x, y):
            def dispatch(m):
                if m == 0:
                    return x
                elif m == 1:
                    return y
                else:
                    raise ValueError("Argument not 0 or 1")
            return dispatch
        case Isempty(lst):
            if lst is None:
                return True
            return False
        case Head(lst):
            if eval(Isempty(lst),environment):
                raise ValueError("Empty list has no head")
            return lst(0)
        case Tail(lst):
            if eval(Isempty(lst),environment):
                raise ValueError("Empty list has no tail")
            elif eval(Isempty(lst(1)),environment):
                print(None)
                return
            else:
                lst=eval(lst(1),environment)
                print(eval(Head(lst),environment),' -> ', end=' ')
                return eval(Tail(lst),environment)
        # adding case for print statement
        case Print(exp):
            value = eval(exp, environment)
            print(value)
            return value
        
        # adding case for functions
        case FunCall(parameters, body):
            def function_eval(arguments: List[Value]) -> Value:
                # create a copy of the environment for the function
                # a new environment is created since each function has it's own environment and own set of local variables 
                function_environment = environment.copy()

                # mapping the arguments to the parameter names
                for name, value in zip(parameters, arguments):
                    function_environment[name] = value

                # evaluating the function in the new environment
                result = None
                for expr in body:
                    result = eval(expr, function_environment)
                return result
            
            return function_eval
        
        # adding case for function calls
        case FunCall.call(name, arguments):
            function = eval(Variable(name), environment)          
            # evaluating the arguments
            evaluated_arguments = [eval(arg, environment) for arg in arguments]
            # calling the function with the evaluated arguments
            return function(evaluated_arguments)
        case _:
            raise InvalidProgram()
def boolify(s: AST):
    e = eval(s)
    if typeof(e) == "type Fraction":
        return BoolLiteral(True) if e > 0 else BoolLiteral(False)
    elif typeof(e) == "type string":
        return BoolLiteral(True) if len(e) > 0 else BoolLiteral(False)
    else:
        return s
    
environ = Environment()


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current_token = None
        self.position = -1
        self.advance()

    def advance(self):
        self.position += 1
        self.current_token = (
            self.tokens[self.position] if self.position < len(self.tokens) else None
        )

    def parse_atom(self):
        match self.current_token:
            case Num(value):
                self.advance()
                return NumLiteral(value)
            case Identifier(name):
                self.advance()
                return Var(name=name)
            case Bool(value):
                self.advance()
                return BoolLiteral(value)
            case Paranthesis("("):
                return self.parse_paran()
            case String(string):
                return Variable(string)
            case Keyword("let"):
                return self.parse_let()
        if type(self.current_token) == List:
            return Parser(self.current_token).parse_expr()
    
    def parse_let(self):
        self.advance()
        var=self.parse_atom()
        self.advance()
        left=self.parse_bool()
        self.advance()
        right=self.parse_bool()
        return Let(var,left,right)

    def parse_exp(self):
        left = self.parse_atom()
        while True:
            match self.current_token:
                case Operator(value) if value == "^":
                    self.advance()
                    right = self.parse_atom()
                    left = BinOp(value, left, right)
                case _:
                    break
        return left

    def parse_mul(self):
        left = self.parse_exp()
        while True:
            match self.current_token:
                case Operator(value) if value in "*/%":
                    self.advance()
                    right = self.parse_exp()
                    left = BinOp(value, left, right)
                case _:
                    break
        return left

    def parse_add(self):
        left = self.parse_mul()
        while True:
            match self.current_token:
                case Operator(value) if value in "+-":
                    self.advance()
                    right = self.parse_mul()
                    left = BinOp(value, left, right)
                case _:
                    break
        return left

    def parse_bool(self):
        left = self.parse_add()
        match self.current_token:
            case Operator(value) if value in "> < == != <= >=".split():
                self.advance()
                right = self.parse_add()
                left = BinOp(value, left, right)
        return left

    def parse_assign(self):
        left = self.parse_bool()
        match self.current_token:
            case Operator("="):
                self.advance()
                right = self.parse_bool()
                left = BinOp("=", left, right)
        return left

    def parse_ifelse(self):
        IF = self.parse_bool()
        self.advance()
        COND = self.parse_expr()
        THEN = self.parse_bool()
        self.advance()
        TRUE = Parser(self.current_token).splitter()
        self.advance()
        ELSE = self.parse_bool()
        self.advance()
        FALSE = Parser(self.current_token).splitter()
        return If(COND, TRUE, FALSE)

    def parse_print(self):
        self.advance()
        return Print(self.parse_paran())

    def parse_loop(self):
        self.advance()
        var = Var(name=self.current_token.word, value=None)
        self.advance()
        assert self.current_token == Keyword("from")
        self.advance()
        low = self.parse_atom()
        assert self.current_token == Keyword("to")
        self.advance()
        high = self.parse_atom()
        task = Parser(self.current_token).splitter()
        return ForLoop(var, low, high, task)

    def parse_expr(self):
        match self.current_token:
            case Keyword(word):
                match word:
                    case "if":
                        return self.parse_ifelse()
                    case "print":
                        return self.parse_print()
                    case "for":
                        return self.parse_loop()
                    case "while":
                        return self.parse_whileloop()
                    case "def":
                        return self.parse_function()
                    case "let":
                        return self.parse_let()
            case Identifier(name):
                return self.parse_assign()
            case _:
                return self.parse_bool()

    def parse_function(self):
        self.advance()
        pass

    def parse_whileloop(self):
        self.advance()
        cond = self.parse_bool()
        task = Parser(self.current_token).splitter()
        return WhileLoop(cond, task)

    def parse_paran(self):
        assert self.current_token == Paranthesis("(")
        self.advance()
        token_temp = []
        num = 1
        while True:
            s = self.current_token
            if s == Paranthesis("("):
                num += 1
            if s == Paranthesis(")"):
                num = num - 1
            if num == 0:
                break
            token_temp.append(s)
            self.advance()
        self.advance()
        return Parser(token_temp).parse_expr()

    def splitter(self):
        ast = []
        temp = self.tokens[:]
        while len(temp) > 0:
            ast.append(Parser(temp[: temp.index(Delimiter(";"))]).parse_expr())
            temp = temp[temp.index(Delimiter(";")) + 1 :]
        return ast

    def main(self):
        asts = self.splitter()
        for ast in asts:
            if type(ast) == List:
                Parser(ast).main()
            else:
                eval(ast, environ)

def test_ForLoop():
    # for loop that sums up the numbers from 1 to 5
    e1 = Variable.make("sum")
    e2 = Variable.make("i")
    ast = ForLoop(
        e2,
        NumLiteral(1),
        NumLiteral(5),
        BinOp("=", e1, BinOp("*", e1, e2))
    )
    environment = Environment()
    environment.add(e1, NumLiteral(1))
    result = eval(ast, environment)
    print(result)
    # assert result == Fraction(120)

def test_concat():
    a=Variable("hello")
    b=Variable("world")
    c=BinOp("+",a,b)
    assert eval(c)== "helloworld"
    
def test_slice(a:Variable):
    Variable.slicing(Variable,a,1,4)

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


def test_function():
    # Test for base case
    base = eval(FunCall(['n'], [If(BinOp("==", Variable('n'), NumLiteral(0)), NumLiteral(1), BinOp("*", Variable('n'), FunCall.call('factorial', [BinOp("-", Variable('n'), NumLiteral(1))])))]) \
                .call([NumLiteral(0)]))
    assert base == 1
    
    # Test for n = 5
    test_1 = eval(FunCall(['n'], [If(BinOp("==", Variable('n'), NumLiteral(0)), NumLiteral(1), BinOp("*", Variable('n'), FunCall.call('factorial', [BinOp("-", Variable('n'), NumLiteral(1))])))]) \
                .call([NumLiteral(5)]))
    assert test_1 == 120

def test_mutarray_eval():
    e1 = NumLiteral(1)
    e2 = NumLiteral(2)
    e3 = NumLiteral(3)
    e4 = MutableArray([e1, e2])
    e5 = Append(e4, e3)
    e6 = Index(e4, NumLiteral(0))
    e7 = Pop(e5)
    e8 = Concat(Concat(e4, e5), MutableArray([]))
    e9 = Assign(e4, NumLiteral(0), NumLiteral(0))


    assert eval(e4) == [1, 2]
    assert eval(e5) == [1, 2, 3]
    assert eval(e6) == 1
    assert eval(Index(e5, NumLiteral(2))) == 3
    assert eval(e7) == [1,2]
    assert eval(e8) == [1, 2, 1, 2, 3]
    assert eval(e9) == [0, 2]

def test_for_list():
    a=1
    b=2
    c=3
    d=4
    e=5
    lst=Cons(a,Cons(b,Cons(c,Cons(d,Cons(e,None)))))
    environment=Environment()
    list=eval(lst,environment) #a list is constructed as 1,2,3,4,5
    print(eval(Head(list),environment)) 
    print(eval(Isempty(list),environment)) 
    eval(Tail(list),environment) # gives tail as 2  ->  3  ->  4  ->  5  ->  None
    list2= eval(list(1),environment)
    print(eval(Head(list2),environment))
    eval(Tail(list2),environment)
    


test_for_list()



s = input()
text = open(s).read()
l = Lexer(text).tokenize()
Parser(l).main()