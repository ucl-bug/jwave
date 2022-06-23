import os
import sys

import pytest
from matplotlib import pyplot as plt

THIS_PATH = os.path.dirname(os.path.realpath(__file__))

TEST_REPORT_DATA = THIS_PATH + "/../docs/test_reports/test_data.txt"
TEST_REPORT_FILE = THIS_PATH + "/../docs/test_reports/test_report.md"

# each test runs on cwd to its temp dir
@pytest.fixture(autouse=True)
def go_to_tmpdir(request):
    # Get the fixture dynamically by its name.
    tmpdir = request.getfixturevalue("tmpdir")
    # ensure local test created packages can be imported
    sys.path.insert(0, str(tmpdir))
    # Chdir only for the duration of the test.
    with tmpdir.as_cwd():
        yield

@pytest.hookimpl()
def pytest_sessionstart(session):
  # Make the test report data file
  if os.path.exists(TEST_REPORT_DATA):
    os.remove(TEST_REPORT_DATA)
  open(TEST_REPORT_DATA, "w").close()

@pytest.hookimpl()
def pytest_sessionfinish(session, exitstatus):
  # Load the test report data
  with open(TEST_REPORT_DATA, "r") as f:
    test_data = f.readlines()
  os.remove(TEST_REPORT_DATA)

  # Make the test report file
  if os.path.exists(TEST_REPORT_FILE):
    os.remove(TEST_REPORT_FILE)
  open(TEST_REPORT_FILE, "w").close()

  results = [
    {"test name": x.split('\t')[0], "Accuracy": float(x.split('\t')[1])} for x in test_data
  ]

  # Sort results by accuracy
  results = sorted(results, key=lambda x: x["Accuracy"], reverse=True)

  results_table = markdown_table(results)

  # Generate a barplot
  plt.figure(figsize=(15, 10))
  plt.grid()

  plt.bar(
    [x["test name"] for x in results],
    [x["Accuracy"] for x in results],
    color="black",
    width=0.8,
  )
  plt.xlabel("Test Name")
  plt.ylabel("Accuracy")
  plt.title("Accuracy of Tests vs k-Wave")
  plt.yscale("log")
  plt.xticks(rotation=45, ha='right')

  # Make sure that the labels fit
  plt.tight_layout()

  # Add an horizontal line at 0.01
  plt.axhline(y=0.01, color="red", linestyle="-.", label="1% error")
  plt.legend()

  # Save the barplot
  plt.savefig(
    THIS_PATH + "/../docs/test_reports/test_accuracy.png",
    dpi=200,
    bbox_inches = "tight")
  plt.close()

  # Generate the markdown file
  with open(TEST_REPORT_FILE, "a") as f:
    f.write("\n\n")
    f.write("# Test Accuracy\n")
    f.write("\n")
    f.write("![Test Accuracy](test_accuracy.png)\n")
    f.write("\n")
    f.write(results_table)
    f.write("\n")


def markdown_table(results):
  # Create the markdown table
  # Adapted from https://github.com/codazoda/tomark/blob/master/tomark/tomark.py

  markdowntable = ""
  markdownheader = '| ' + ' | '.join(map(str, results[0].keys())) + ' |'
  markdownheaderseparator = '|-----' * len(results[0].keys()) + '|'
  markdowntable += markdownheader + '\n'
  markdowntable += markdownheaderseparator + '\n'
  for row in results:
      markdownrow = ""
      for key, col in row.items():
          markdownrow += '| ' + to_string_safe(col) + ' '
      markdowntable += markdownrow + '|' + '\n'
  return markdowntable

def to_string_safe(text):
  if type(text) == str:
    return text
  elif type(text) == float:
    # Exponent notation rounded to 2 decimal places
    return "{:.2e}".format(text)
  elif type(text) == int:
    return str(text)
  else:
    return str(text)

if __name__ == "__main__":
  pass
