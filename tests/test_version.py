import configparser
import os.path

import qfeval_data


def test_version():
    parser = configparser.ConfigParser()
    with open("pyproject.toml", "r") as f:
        lines = f.readlines()[:5]
    parser.read_string("\n".join(lines))
    assert qfeval_data.__version__ == parser["project"]["version"].replace(
        '"', ""
    )
