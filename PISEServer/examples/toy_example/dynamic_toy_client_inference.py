import logging

from pise import server
import pise.monitoring_phase
import pise.hooks_dynamic
from tritondse import callbacks, ProcessState

class ToySendCallSite(pise.hooks_dynamic.SendReceiveCallSite):
    
    def __init__(self) -> None:
        super().__init__()
    
    def set_hook(self, hooks_obj, callback_manager):
        callback_manager.register_function_callback('recv', hooks_obj.RecvHook)
    
    def extract_arguments(self, pstate):
        length = pstate.registers.edx
        buffer = pstate.registers.rsi
        return buffer, length
    
    def get_return_value(self, buffer, length, pstate):
        pstate.register.rax = length
        return length
    
class ToyRecvCallSite(pise.hooks_dynamic.SendReceiveCallSite):
    
    def __init__(self) -> None:
        super().__init__()
    
    def set_hook(self, hooks_obj, callback_manager):
        callback_manager.register_function_callback('send', hooks_obj.SendHook)
        
    def extract_arguments(self, pstate : ProcessState):
        length = pstate.registers.edx
        buffer = pstate.registers.rsi
        return buffer, length
    
    def get_return_value(self, buffer, length, pstate):
        pstate.register.rax = length
        return length
    
    



def main():
    logging.getLogger('pise').setLevel(logging.DEBUG)
    # logging.getLogger('angr').setLevel(logging.INFO)
    query_runner = monitoring_phase.QueryRunner('examples/toy_example/toy_example', [ToySendReceiveCallSite()])
    s = server.Server(query_runner)
    s.listen()


if __name__ == "__main__":
    main()
    

