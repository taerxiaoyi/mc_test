import time

from westlake_sdkpy.core.channel import ChannelPublisher, ChannelFactoryInitialize
from westlake_sdkpy.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from westlake_sdkpy.idl.default import agile_msg_dds__LowState_
from westlake_sdkpy.idl.agile.msg.dds_ import LowState_


class Custom:
    def __init__(self):
        self.counter_ = 0

    def Init(self):
        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    def LowStateHandler(self, msg: LowState_):
        
        self.low_state = msg
        
        print(f"序列号={self.low_state.sequences}")
        

if __name__ == '__main__':
    
    ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()
     
    while True:        
        time.sleep(1)

