from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, List

digit_list = "1234567890"
alphabet_list = "ABCDEFGHIJKLOMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
variable_list = []


@dataclass
class EndOfStream(Exception):
    pass


@dataclass
class Stream:
    string: str
    position: int

    def next_char(self):
        if self.position >= len(self.string):
            raise EndOfStream()
        else:
            self.position += 1
            return self.string[self.position - 1]

    def unget(self):
        assert self.position > 0
        self.position -= 1


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


Token = Num | Bool | Keyword | Identifier | Operator | Paranthesis | Delimiter


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
keywords = "if then else end while do done print for from to".split()
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

    def backtrack(self):
        assert self.pos > 0
        self.pos -= 1
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
                case s if s in "()}{":
                    tokens.append(Paranthesis(s))
                    self.advance()

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

    def slicing(self, start_index: NumLiteral, end_index: NumLiteral):

        if start_index.value > len(self.name) - 1:
            raise IndexError()
        elif end_index.value <= start_index.value:
            raise IndexError()
        elif end_index.value > len(self.name):
            raise IndexError()
        else:
            string_slice = self.name[start_index.value : end_index.value]
            return Variable(string_slice)


@dataclass
class Var:
    name: str
    value: str | int | bool = None


@dataclass
class If:
    cond: "AST"
    true_branch: "AST"
    false_branch: "AST"


# Implementing the BoolLiteral Type
@dataclass
class BoolLiteral:
    value: bool

    def __init__(self, val: bool):
        self.value = val


# implementing the print function
@dataclass
class Print:
    exp: "AST"


@dataclass
class Let:
    var: "AST"
    e1: "AST"
    e2: "AST"


@dataclass
class LetMut:
    var: "AST"
    e1: "AST"
    e2: "AST"


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


@dataclass
class LetFun:
    name: "AST"
    params: List["AST"]
    body: "AST"
    expr: "AST"


@dataclass
class FunCall:
    fn: "AST"
    args: List["AST"]


@dataclass
class ForLoop:
    var: "AST"
    start: "AST"
    end: "AST"
    body: "AST"


AST = (
    NumLiteral
    | BinOp
    | Variable
    | Let
    | LetMut
    | Put
    | Get
    | Seq
    | LetFun
    | FunCall
    | ForLoop
)


@dataclass
class FnObject:
    params: List["AST"]
    body: "AST"


Value = Fraction | FnObject | bool


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
        self.envs.pop()

    def add(self, name, value):
        assert name not in self.envs[-1]
        self.envs[-1][name] = value

    def get(self, name):
        for env in reversed(self.envs):
            if name in env:
                return env[name].value
        raise KeyError()

    def update(self, name, value):
        for env in reversed(self.envs):
            if name in env:
                env[name] = value
                return env[name].value
        self.add(name, value)
        # raise KeyError()


def resolve(program: AST, environment: Environment = None) -> AST:
    if environment is None:
        environment = Environment()

    def resolve_(program: AST) -> AST:
        return resolve(program, environment)

    match program:
        case NumLiteral(_) as N:
            return N
        case Variable(name):
            return environment.get(name)
        case Let(Variable(name) as v, e1, e2):
            re1 = resolve_(e1)
            environment.enter_scope()
            environment.add(name, v)
            re2 = resolve_(e2)
            environment.exit_scope()
            return Let(v, re1, re2)
        case LetFun(Variable(name) as v, params, body, expr):
            environment.enter_scope()
            environment.add(name, v)
            environment.enter_scope()
            for param in params:
                environment.add(param.name, param)
            rbody = resolve_(body)
            environment.exit_scope()
            rexpr = resolve_(expr)
            environment.exit_scope()
            return LetFun(v, params, rbody, rexpr)
        case FunCall(fn, args):
            rfn = resolve_(fn)
            rargs = []
            for arg in args:
                rargs.append(resolve_(arg))
            return FunCall(rfn, rargs)


def eval(program: AST, environment: Mapping[str, Value] = None) -> Value:
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
        case Let(Variable(name), e1, e2):
            v1 = eval(e1, environment)
            return eval(e2, environment | {name: v1})
        case Var(name, value):
            for variable in variable_list:
                if variable.name == name:
                    return variable.value
            raise InvalidProgram("reference before assignment")
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
            return eval(left) % eval(right)
        case If(cond, true_branch, false_branch):
            if eval(cond, environment):
                return eval(true_branch, environment)
            else:
                return eval(false_branch, environment)
        case BinOp("^", left, right):
            return eval(left) ** eval(right)
        case BinOp("!=", left, right):
            return eval(left, environment) != eval(right, environment)
        case BinOp("<=", left, right):
            return eval(left, environment) <= eval(right, environment)
        case BinOp(">=", left, right):
            return eval(left, environment) >= eval(right, environment)
        case BinOp("=", left, right):
            right_eval = eval(right)
            match left:
                case Var(name, value):
                    left.value = right_eval
                    flag = 1
                    for var in variable_list:
                        if var.name == left.name:
                            var.value = left.value
                            flag = 0
                    if flag:
                        variable_list.append(left)
                case _:
                    raise InvalidProgram()
        case ForLoop(var, start, end, body):
            start_val = eval(start)
            end_val = eval(end)
            for j in range(int(start_val), int(end_val) + 1):
                eval(BinOp("=", var, NumLiteral(Fraction(j))))
                eval(body)
            return

        # adding case for print statement
        case Print(exp):
            value = eval(exp, environment)
            print(value)
            return value
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
            case Paranthesis("{"):
                return self.parse_curly()
        if type(self.current_token) == list:
            return Parser(self.current_token).parse_expr()

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
        TRUE = self.parse_expr()
        ELSE = self.parse_bool()
        self.advance()
        FALSE = self.parse_expr()
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
        task = self.parse_curly()
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
            case Identifier(name):
                return self.parse_assign()
            case _:
                return self.parse_bool()

    def parse_curly(self):
        assert self.current_token == Paranthesis("{")
        self.advance()
        token_temp = []
        num = 1
        while True:
            s = self.current_token
            if s == Paranthesis("{"):
                num += 1
            if s == Paranthesis("}"):
                num = num - 1
            if num == 0:
                break
            token_temp.append(s)
            self.advance()
        self.advance()
        return Parser(token_temp).parse_expr()

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
            eval(ast)


s = input()
text = open(s).read()
l = Lexer(text).tokenize()
Parser(l).main()
