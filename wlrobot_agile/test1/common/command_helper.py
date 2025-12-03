from westlake_sdkpy.idl.agile.msg.dds_ import LowCmd_

# 预留接口，目前不支持模式选择
class MotorMode:
    PR = 0             # Series Control for Pitch/Roll Joints 
    AB = 1             # Parallel Control for A/B Joints


def create_damping_cmd(cmd: LowCmd_):
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].mode = 0x02
        cmd.motor_cmd[i].pos = 0
        cmd.motor_cmd[i].w = 0
        cmd.motor_cmd[i].t = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 1         # 未知，根据自己的电机实际值来填
  

def create_zero_cmd(cmd: LowCmd_):
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].mode = 0x02
        cmd.motor_cmd[i].pos = 0
        cmd.motor_cmd[i].w = 0
        cmd.motor_cmd[i].t = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 0        


def init_cmd_agile(cmd: LowCmd_, mode_machine: int, mode_pr: int):
    cmd.mode_machine = mode_machine
    cmd.mode_pr = mode_pr
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].mode = 0x02
        cmd.motor_cmd[i].pos = 0
        cmd.motor_cmd[i].w = 0
        cmd.motor_cmd[i].t = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 0  



