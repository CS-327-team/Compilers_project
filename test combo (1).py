
def test_for_let():
    e1 = NumLiteral(4)
    e2 = NumLiteral(7)
    e3 = NumLiteral(9)
    e4 = NumLiteral(5)
    e5 = BinOp("+", e2, e3)
    e6 = BinOp("/", e5, e4)
    e7 = BinOp("-", e1, e6)
    
    # Create a Let expression
    let_expr = Let(Variable('x'), e1, e7)
    
    # Create a ForLoop expression
    for_loop_expr = ForLoop(Variable('i'), NumLiteral(1), NumLiteral(5), [BinOp("=", Var('x'), NumLiteral(Fraction(2)))])

    # Evaluate the combined expression
    result = eval(BinOp("*", let_expr, for_loop_expr), None)
    
    # Expected result: 72
    assert result == Fraction(72), f"Expected result: 72, Actual result: {result}"
#test_let()


def test_for_loop_with_if_else():
    e1 = NumLiteral(1)
    e2 = NumLiteral(5)
    e3 = BinOp("+", Variable("i"), e1)
    e4 = BinOp("<=", Variable("i"), e2)
    e5 = BinOp("*", Variable("result"), Variable("i"))
    result = Variable("result")
    i = Variable("i")
    environment = Environment()
    environment.enter_scope()
    eval(Let(i, NumLiteral(1)), environment)
    eval(Let(result, NumLiteral(1)), environment)
    e6 = ForLoop(Variable("i"), e1, e2, Put(result, e5))
    e7 = If(BinOp(">", result, NumLiteral(10)), result, e6)
    eval(e7, environment)
    assert environment["result"] == NumLiteral(120)

def test_while_loop_with_if_else():
    e1 = NumLiteral(1)
    e2 = NumLiteral(5)
    e3 = BinOp("<=", Variable("i"), e2)
    e4 = BinOp("+", Variable("i"), e1)
    e5 = BinOp("*", Variable("result"), Variable("i"))
    e6 = BinOp("=", Variable("i"), e4)
    e7 = BinOp("=", Variable("result"), e5)
    e8 = If(BinOp(">", Variable("i"), NumLiteral(3)), e7, e6)
    e9 = While(e3, [e8])
    environment = {"i": e1, "result": NumLiteral(1)}
    eval(e9, environment)
    assert environment["result"] == NumLiteral(24)

def test_while_if_else_list():
    # creating a list of numbers
    e1 = NumLiteral(1)
    e2 = NumLiteral(3)
    e3 = NumLiteral(5)
    e4 = NumLiteral(6)
    e5 = NumLiteral(8)
    e6 = Cons(e1, Cons(e2, Cons(e3, Cons(e4, Cons(e5, NumLiteral(0))))))

    # setting up while loop to remove odd numbers from list
    e7 = BinOp("<", Variable("i"), Length(e6))   # if false, the while loop terminates
    # if the number is even, keep it in the list, otherwise a list with 2 0s is created
    e8 = If(BinOp("==", BinOp("%", Index(e6, Variable("i")), NumLiteral(2)), NumLiteral(0)),
            Index(e6, Variable("i")),
            Cons(NumLiteral(0), NumLiteral(0)))
    # the statement i = i + 1
    e9 = BinOp("+", Variable("i"), NumLiteral(1))
    e10 = BinOp("=", Variable("i"), e9)
    e11 = BinOp("=", Index(e6, Variable("j")), e8)
    e12 = If(BinOp("==", BinOp("%", Index(e6, Variable("i")), NumLiteral(2)), NumLiteral(0)), e11, e10)
    e13 = While(e7, [e12])

    # evaluating while loop with initial environment
    environment = {"i": NumLiteral(1), "j": NumLiteral(1)}
    eval(e13, environment)

    assert eval(e6) == [6, 8, 0]

def test_for_loop_with_if_else_and_list():
    e1 = NumLiteral(1)
    e2 = NumLiteral(5)
    e3 = BinOp("+", Variable("i"), e1)  # i = i + 1
    e4 = BinOp("<=", Variable("i"), e2) # if false, the for loop terminates
    e5 = BinOp("*", Variable("result"), Variable("i")) # result = result * i
    result = Variable("result")
    i = Variable("i")
    environment = Environment()
    # for loop multiples numbers 1 to 5 and stores the result in a list with 2 0s
    environment.enter_scope()
    eval(Let(i, NumLiteral(1)), environment)
    eval(Let(result, NumLiteral(1)), environment)
    e6 = ForLoop(Variable("i"), e1, e2, Put(result, e5))
    e7 = If(BinOp(">", result, NumLiteral(10)),
            Cons(NumLiteral(10), Cons(result, Cons(NumLiteral(0), NumLiteral(0)))),
            Cons(result, Cons(NumLiteral(0), NumLiteral(0))))
    eval(e7, environment)
    assert environment["result"] == NumLiteral(120)
    assert eval(e7) == [120, 0, 0]

def test_while_loop_with_if_and_mutArray():
    e1 = NumLiteral(1)
    e2 = NumLiteral(10)
    e3 = BinOp("<=", Variable("num"), e2) # while loop terminates if false
    e4 = BinOp("mod", Variable("num"), NumLiteral(2)) # creates the remainder of num/2 to check if the number is even
    e5 = BinOp("+", Variable("sum"), Variable("num")) # sum = sum + num
    e6 = If(e4, Append(Variable("even_nums"), Variable("num"))) # if the number is even, append it to the list  "even_nums"
    e7 = Assign(Variable("num"), BinOp("+", Variable("num"), NumLiteral(1))) # num = num + 1
    e8 = While(e3, [e6, e7]) # while loop
    e9 = MutableArray([]) # creating an empty array
    e10 = Assign(Variable("even_nums"), NumLiteral(1), e9) 
    environment = {"num": e1, "sum": NumLiteral(0)} 
    eval(e10, environment) 
    eval(e8, environment)
    assert environment["sum"] == NumLiteral(30) # sum of all even numbers from 1 to 10
    assert eval(e9) == [2, 4, 6, 8, 10]   # all even number from 1 to 10

def test_for_loop_with_if_and_mutArray():
    e1 = NumLiteral(1)
    e2 = NumLiteral(10)
    e3 = MutableArray([])
    # appending odd numbers to the mutable array
    e4 = ForLoop(Variable("i"), e1, e2, If(BinOp("%", Variable("i"), NumLiteral(2)), Append(e3, Variable("i"))))
    
    # adding even numbers from the mutable array to a sum variable using if-else statement
    e5 = NumLiteral(0)
    e6 = ForLoop(Variable("i"), e1, e2, If(BinOp("%", Variable("i"), NumLiteral(2)), Assign(Variable("sum"), BinOp("+", Variable("sum"), Variable("i"))), None))
    
    # changing the first element of the mutable array to 0 
    e7 = Assign(Index(e3, NumLiteral(0)), NumLiteral(0))
    # popping the last element
    e8 = Pop(e3)
    
    # concatenating the mutable array with itself and assign to a new variable
    e9 = Concat(e3, e3)
    e10 = Assign(Variable("new_array"), e3, e9)
    
    # create environment and evaluate expressions
    environment = Environment()
    eval(e4, environment)
    eval(e6, environment)
    eval(e7, environment)
    eval(e8, environment)
    eval(e10, environment)
    
    # check assertions
    assert environment["sum"] == NumLiteral(20)
    assert eval(e3) == [0, 3, 5, 7, 9]
    assert eval(e10) == [0, 3, 5, 7, 9, 0, 3, 5, 7, 9]

def test_for_loop_if_let_eval():
    a = Variable("a")
    result = Variable("result")
    i = Variable("i")
    e1 = NumLiteral(1)
    e2 = NumLiteral(5)
    e3 = BinOp("+", i, e1)
    e4 = BinOp("<=", i, e2)
    e5 = BinOp("*", result, i)
    # if i > 3, result = result * i
    e6 = ForLoop(i, e1, e2, Let(result, If(BinOp(">", i, NumLiteral(3)), e5, result)))
    environment = Environment()
    environment.enter_scope()
    eval(Let(a, NumLiteral(2)), environment)
    eval(e6, environment)
    assert environment[result.name] == NumLiteral(48)
    e7 = Let(a, BinOp("+", a, NumLiteral(3)), e6)
    eval(e7, environment)
    assert environment[result.name] == NumLiteral(720)

def test_while_concat():
    a = Variable("hello")
    b = Variable("world")
    c = BinOp("+", a, b)
    assert eval(c) == "helloworld"

    e1 = NumLiteral(1)
    e2 = NumLiteral(5)
    e3 = BinOp("<=", Variable("i"), e2)  # while loop terminates if false
    e4 = BinOp("+", Variable("i"), e1) # i + 1
    e5 = BinOp("*", Variable("result"), Variable("i")) # result * i
    e6 = BinOp("=", Variable("i"), e4) # i = i + 1
    e7 = BinOp("=", Variable("result"), e5) # result = result * i
    e8 = While(e3, [e7, e6]) # while loop concatenates the string "helloworld" 5 times
    environment = {"i": e1, "result": c} 
    eval(e8, environment)
    assert environment["result"] == "helloworldhelloworldhelloworldhelloworldhelloworld"

def test_for_loop_concat():
    a = Variable("hello")
    b = Variable("world")
    e1 = NumLiteral(1)
    e2 = NumLiteral(5) 
    e3 = BinOp("+", Variable("i"), e1)  # i + 1
    e4 = BinOp("<=", Variable("i"), e2)  # while loop terminates if false
    e5 = BinOp("*", a, Variable("i")) # result * i
    e6 = BinOp("+", a, b) 
    result = Variable("result") 
    i = Variable("i")
    environment = Environment()
    environment.enter_scope()
    eval(Let(i, NumLiteral(1)), environment)
    eval(Let(result, e6), environment)
    e7 = ForLoop(Variable("i"), e1, e2, Put(result, e5))  # for loop concatenates the string "hello" 5 times
    eval(e7, environment)
    assert eval(result, environment) == "hellohellohellohellohello" + "helloworld"

def test_if_else_concat_eval():
    a = Variable("hello")
    b = Variable("world")
    c = BinOp("+", a, b)
    d = BinOp(">", NumLiteral(9), NumLiteral(7))
    e = If(d, c, a)  # if 9 > 7, return "helloworld", else return "hello"
    assert eval(e) == "helloworld"

