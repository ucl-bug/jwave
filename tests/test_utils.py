from jwave.utils import is_numeric


def test_is_numeric():
  assert is_numeric(1)
  assert is_numeric(1.0)
  assert is_numeric(1j)
  assert not is_numeric("1")
