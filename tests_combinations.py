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


