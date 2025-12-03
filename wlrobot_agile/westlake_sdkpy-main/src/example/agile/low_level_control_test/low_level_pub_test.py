import time

from westlake_sdkpy.core.channel import ChannelPublisher, ChannelFactoryInitialize
from westlake_sdkpy.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from westlake_sdkpy.idl.default import agile_msg_dds__LowCmd_
from westlake_sdkpy.idl.agile.msg.dds_ import LowCmd_

from westlake_sdkpy.idl.default import agile_msg_dds__LowState_
from westlake_sdkpy.idl.agile.msg.dds_ import LowState_
from westlake_sdkpy.utils.thread import RecurrentThread


class Custom:
    def __init__(self):
        self.control_dt_ = 0.002  # [2ms]
        self.lowstate = agile_msg_dds__LowState_()

    def Init(self):
        # create publisher 
        self.lowstate_publisher_ = ChannelPublisher("rt/lowstate", LowState_)
        self.lowstate_publisher_.Init()

    def Start(self):

        self.lowstateWriteThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.LowStateWrite, name="state"
        )
       
        self.lowstateWriteThreadPtr.Start()


    def LowStateWrite(self):  
        
        self.lowstate.sequences += 1
        
        self.lowstate_publisher_.Write(self.lowstate)


if __name__ == '__main__':

    ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()
    custom.Start()

    while True:        
        time.sleep(1)