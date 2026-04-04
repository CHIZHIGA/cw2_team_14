# generated from rosidl_generator_py/resource/_idl.py.em
# with input from cw2_world_spawner:srv/Task3Service.idl
# generated code does not contain a copyright notice


# Import statements for member types

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_Task3Service_Request(type):
    """Metaclass of message 'Task3Service_Request'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('cw2_world_spawner')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'cw2_world_spawner.srv.Task3Service_Request')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__task3_service__request
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__task3_service__request
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__task3_service__request
            cls._TYPE_SUPPORT = module.type_support_msg__srv__task3_service__request
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__task3_service__request

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class Task3Service_Request(metaclass=Metaclass_Task3Service_Request):
    """Message class 'Task3Service_Request'."""

    __slots__ = [
    ]

    _fields_and_field_types = {
    }

    SLOT_TYPES = (
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)


# Import statements for member types

# Member 'most_common_shape_vector'
import array  # noqa: E402, I100

import builtins  # noqa: E402, I100

# already imported above
# import rosidl_parser.definition


class Metaclass_Task3Service_Response(type):
    """Metaclass of message 'Task3Service_Response'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('cw2_world_spawner')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'cw2_world_spawner.srv.Task3Service_Response')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__srv__task3_service__response
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__srv__task3_service__response
            cls._CONVERT_TO_PY = module.convert_to_py_msg__srv__task3_service__response
            cls._TYPE_SUPPORT = module.type_support_msg__srv__task3_service__response
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__srv__task3_service__response

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class Task3Service_Response(metaclass=Metaclass_Task3Service_Response):
    """Message class 'Task3Service_Response'."""

    __slots__ = [
        '_total_num_shapes',
        '_num_most_common_shape',
        '_most_common_shape_vector',
    ]

    _fields_and_field_types = {
        'total_num_shapes': 'int64',
        'num_most_common_shape': 'int64',
        'most_common_shape_vector': 'sequence<int64>',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.BasicType('int64'),  # noqa: E501
        rosidl_parser.definition.BasicType('int64'),  # noqa: E501
        rosidl_parser.definition.UnboundedSequence(rosidl_parser.definition.BasicType('int64')),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.total_num_shapes = kwargs.get('total_num_shapes', int())
        self.num_most_common_shape = kwargs.get('num_most_common_shape', int())
        self.most_common_shape_vector = array.array('q', kwargs.get('most_common_shape_vector', []))

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.total_num_shapes != other.total_num_shapes:
            return False
        if self.num_most_common_shape != other.num_most_common_shape:
            return False
        if self.most_common_shape_vector != other.most_common_shape_vector:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def total_num_shapes(self):
        """Message field 'total_num_shapes'."""
        return self._total_num_shapes

    @total_num_shapes.setter
    def total_num_shapes(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'total_num_shapes' field must be of type 'int'"
            assert value >= -9223372036854775808 and value < 9223372036854775808, \
                "The 'total_num_shapes' field must be an integer in [-9223372036854775808, 9223372036854775807]"
        self._total_num_shapes = value

    @builtins.property
    def num_most_common_shape(self):
        """Message field 'num_most_common_shape'."""
        return self._num_most_common_shape

    @num_most_common_shape.setter
    def num_most_common_shape(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'num_most_common_shape' field must be of type 'int'"
            assert value >= -9223372036854775808 and value < 9223372036854775808, \
                "The 'num_most_common_shape' field must be an integer in [-9223372036854775808, 9223372036854775807]"
        self._num_most_common_shape = value

    @builtins.property
    def most_common_shape_vector(self):
        """Message field 'most_common_shape_vector'."""
        return self._most_common_shape_vector

    @most_common_shape_vector.setter
    def most_common_shape_vector(self, value):
        if isinstance(value, array.array):
            assert value.typecode == 'q', \
                "The 'most_common_shape_vector' array.array() must have the type code of 'q'"
            self._most_common_shape_vector = value
            return
        if __debug__:
            from collections.abc import Sequence
            from collections.abc import Set
            from collections import UserList
            from collections import UserString
            assert \
                ((isinstance(value, Sequence) or
                  isinstance(value, Set) or
                  isinstance(value, UserList)) and
                 not isinstance(value, str) and
                 not isinstance(value, UserString) and
                 all(isinstance(v, int) for v in value) and
                 all(val >= -9223372036854775808 and val < 9223372036854775808 for val in value)), \
                "The 'most_common_shape_vector' field must be a set or sequence and each value of type 'int' and each integer in [-9223372036854775808, 9223372036854775807]"
        self._most_common_shape_vector = array.array('q', value)


class Metaclass_Task3Service(type):
    """Metaclass of service 'Task3Service'."""

    _TYPE_SUPPORT = None

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('cw2_world_spawner')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'cw2_world_spawner.srv.Task3Service')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._TYPE_SUPPORT = module.type_support_srv__srv__task3_service

            from cw2_world_spawner.srv import _task3_service
            if _task3_service.Metaclass_Task3Service_Request._TYPE_SUPPORT is None:
                _task3_service.Metaclass_Task3Service_Request.__import_type_support__()
            if _task3_service.Metaclass_Task3Service_Response._TYPE_SUPPORT is None:
                _task3_service.Metaclass_Task3Service_Response.__import_type_support__()


class Task3Service(metaclass=Metaclass_Task3Service):
    from cw2_world_spawner.srv._task3_service import Task3Service_Request as Request
    from cw2_world_spawner.srv._task3_service import Task3Service_Response as Response

    def __init__(self):
        raise NotImplementedError('Service classes can not be instantiated')
