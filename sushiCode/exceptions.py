#!/usr/bin/env python
# encoding: utf-8

class Error(Exception):
    pass
class IllegalMassSizeError(Error):
    """Exception when size of mass vector is not 2 ^n"""
    def __init__(self, message):
        super(IllegalMassSizeError, self).__init__(message)

class IllegalMassVlaueError(Error):
    """Exception when the value of a mass is over 1 or negative"""
    def __init__(self, message):
        super(IllegalMassVlaueError, self).__init__(message)

class IllegalMatrixShapeError(Error):
    """Exception when two matrix are of the differet shape"""
    def __init__(self, message):
        super(IllegalMatrixShapeError, self).__init__(message)

class IllegalItemNumberError(Error):
    """Exception when sequence size is larger than items number"""
    def __init__(self, message):
        super(IllegalItemNumberError, self).__init__(message)

class MatrixShapeMatchError(Error):
    "Exception when the given matrix lists do not have the same shape"
    def __init__(self, message):
        super(MatrixShapeMatchError, self).__init__(message)

class IllegalParameterError(Error):
    "Exceptino when the given parameter in a method is not supported"
    def __init__(self, message):
        super(IllegalParameterError, self).__init__(message)
