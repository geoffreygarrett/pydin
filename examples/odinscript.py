import pydin


# types in lib
# pydin.int32
# pydin.int64
# pydin.float32
# pydin.float64
# pydin.state
# pydin.variable

@pydin.jit
def test(a: int, b: int) -> int:
    c: int = a + b
    return c ** 2


# @pydin.jit
# def reward(s: pydin.state) -> pydin.float64:
#     return s.x ** 2 + s.y ** 2
#
#
# @pydin.jit
# def policy(s: pydin.state) -> pydin.action:
#     return s.x + s.y


if __name__ == '__main__':
    print(test(1, 2))
    print(test.__dir__())  # 'func_name', 'args', 'return_type', 'body',
    print(test.__annotations__)  # {'a': <class 'int'>, 'b': <class 'int'>, 'c': <class 'int'>, 'return': <class 'int'>}
    print(
        test.__code__)  # <code object test at 0x0000020F4F6B1C00, file "C:\Users\james\PycharmProjects\pydin\examples\odinscript.py", line 25>
    print(test.func_name)  # test
    print(test.args)  # [('a', 'int'), ('b', 'int')]
    print(test.return_type)  # int
