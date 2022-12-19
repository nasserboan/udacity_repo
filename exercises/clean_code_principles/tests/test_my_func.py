from app.my_func import func1
from pytest import fixture

@fixture
def print_fixture():
    print('Executando fixture')
    return 

def test_func1(print_fixture):
    resultado = func1(2,10)
    assert resultado == 8

def test_negative_func1(print_fixture):
    resultado = func1(10,2)
    assert resultado == 0
