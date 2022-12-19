"""
Module for holding and computing the diff_parameter function.

Author: Nasser Boan
Date: December, 2022
"""

import logging
import sys

logging.basicConfig(
    filename='./results.log',
    level=logging.INFO,
    filemode='a',
    datefmt="%Y-%m-%d %H:%M:%S",
    format='%(asctime)s - %(levelname)s - %(message)s')


def func1(arg1, arg2):
    """
    Function for returning the non-negative differentece between parameters.

    Args:
    _____
        * arg1 : [int, float] parameter 1.
        * arg2 : [int, float] parameter 2.

    Returns:
    _____
        * result: [int, float] difference between paramenter 2 and
                  paramenter 1. If parameter 2 is higher than parameter 1 the
                  function will return zero.

    """

    if arg1 > arg2:
        logging.warning(
            " arg2(%s) greater than arg1(%s), returning zero.",
            arg2,
            arg1)
        result = 0
        return result

    logging.info("SUCCESS: arguments (%s,%s) recieved.", arg1, arg2)
    result = arg2 - arg1
    return result


if __name__ == "__main__":
    ARG1_STR, ARG2_STR = sys.argv[1], sys.argv[2]

    ARG1_NUM, ARG2_NUM = int(ARG1_STR), int(ARG2_STR)

    func1(ARG1_NUM, ARG2_NUM)
