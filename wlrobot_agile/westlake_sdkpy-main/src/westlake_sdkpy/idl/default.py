from .agile.msg.dds_ import *

def agile_msg_dds__IMUState_():
    return IMUState_([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, 0)

def agile_msg_dds__BmsState_():
    return BmsState_([0, 0, 0], 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0)

def agile_msg_dds__MotorCmd_():
    return MotorCmd_(0, 0.0, 0.0, 0.0, 0.0, 0.0)

def agile_msg_dds__LowCmd_():
    return LowCmd_(0, 0, 0, [agile_msg_dds__MotorCmd_() for i in range(32)], 0)

def agile_msg_dds__MotorState_():
    return MotorState_(0, 0, [0, 0], 0.0, 0.0, 0.0, 0)

def agile_msg_dds__LowState_():
    return LowState_(0, 0, 0, [agile_msg_dds__MotorState_() for i in range(32)], agile_msg_dds__IMUState_(),
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0)
