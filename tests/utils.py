from .conftest import TEST_REPORT_DATA


def log_accuracy(name, result):
  """
  Logs the accuracy of a test

  Args:
    name: name of the test
    result: result of the test
  """
  with open(TEST_REPORT_DATA, "a") as f:
    f.write(f"{name}\t{result}\n")
  return
