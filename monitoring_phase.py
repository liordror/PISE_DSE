import copy
import logging

import tritondse
from tritondse import SymbolicExecutor, ProcessState, callbacks
from tritondse import SymbolicExecutor, Config, Seed, CompositeData
from tritondse import Program
import triton

from pise.entities import MessageTypeSymbol

NUM_SOLUTIONS = 10
logger = logging.getLogger(__name__)


class QueryRunner:
    def __init__(self, file, callsites_to_monitor):
        self.file = file
        self.project  = Program(file)
        self.mode = None
        self.callsites_to_monitor = callsites_to_monitor
        self.set_membership_hooks()
        #self.probing_cache = ProbingCache()
        self.state_stack = []
        self.inputs = None
        self.monitoring_result = False
    
    def sendHook(self, se : SymbolicExecutor, pstate : ProcessState, addr : int):
        #we are in monitoring phase
        if pstate.monitoring_position < len(self.inputs):
            if self.inputs[pstate.monitoring_phase].type == 'SEND':
                msg = pstate.memory.read_string(pstate.get_argument_value(0)) ## from where does it read?
                length = len(msg)
                for (byte, val) in self.inputs[pstate.monitoring_position].predicate.items():  ## question: are we sure its the sintax?
                    if byte >= length:
                        continue 
                    if msg[byte] != val:
                        self.monitoring_result = (self.monitoring_result or False)  ## question: maybe problematic
                        se.stop_exploration()
                        #new path from stack
                pstate.monitoring_position += 1
            else:
                self.monitoring_result = (self.monitoring_result or False)
                se.stop_exploration()
                #new path from stack
        else:
            pass
    
    def recvHook(self, se : SymbolicExecutor, pstate : ProcessState, addr : int):
        if pstate.monitoring_position < len(self.inputs):
            if self.inputs[pstate.monitoring_phase].type == 'RECV':
                msg_addr = pstate.get_argument_value(0) ## what is this different from send?
                #other possebility (not sure) - compute a msg that answers pradicate. 
                for (byte, val) in self.inputs[pstate.monitoring_position].predicate.items():
                    ## what about checking the length of the message?
                    byte_addr = msg_addr + byte
                    sym_mem = pstate.read_symbolic_memory_byte(byte_addr) ## what is there to read? is it just for the condition later?
                    pstate.write_symbolic_memory_byte(byte_addr, sym_mem == val)
                pstate.monitoring_position += 1
            else:
                self.monitoring_result = (self.monitoring_result or False)
                se.stop_exploration()
                #new path from stack
        else:
            pass
    
    def branchHook(self, se : SymbolicExecutor, pstate: ProcessState, opcode: triton.OPCODE):
        self.state_stack.append(pstate)
        #duplicate pstate including new fields

    def executionEnded(self, se: SymbolicExecutor, pstate: ProcessState):
        if len(self.state_stack) > 0:
            new_pstate = self.state_stack.pop()
            
            config = Config()
            seed = Seed(CompositeData())
            
            new_se = SymbolicExecutor(config, seed)
            new_se.load_process(new_pstate)
            new_se.run()
        else:
            return
    
    def set_membership_hooks(self):
        if self.mode == 'membership':
            return
        logger.info('Setting hooks')
        # setting hooks for send, recv
        for callsite in self.callsites_to_monitor:
            callsite.set_hook(self.project)
        self.mode = 'membership'
        #setting hooks on all branch opcodes
        branch_opcodes = [triton.OPCODE.X86.JAE, triton.OPCODE.X86.JA, triton.OPCODE.X86.JBE, triton.OPCODE.X86.JB, 
                          triton.OPCODE.X86.JCXZ, triton.OPCODE.X86.JECXZ, triton.OPCODE.X86.JE, triton.OPCODE.X86.JGE, 
                          triton.OPCODE.X86.JG, triton.OPCODE.X86.JLE, triton.OPCODE.X86.JL, triton.OPCODE.X86.JNE, 
                          triton.OPCODE.X86.JNO, triton.OPCODE.X86.JNP, triton.OPCODE.X86.JNS, triton.OPCODE.X86.JO, 
                          triton.OPCODE.X86.JP, triton.OPCODE.X86.JRCXZ, triton.OPCODE.X86.JS]
        # path_constraint = self.pstate.last_branch_constraint
        # maybe we can switch the stack to this command and we dont need hook_branch
        for mnemonic in branch_opcodes:
             callbacks.CallbackManager.register_mnemonic_callback(callbacks.BEFORE, mnemonic, self.branchHook)
        callbacks.CallbackManager.register_post_execution_callback(self.executionEnded) ## we didnt change that?
        
    def membership_step_by_step(self, inputs):
        self.inputs = inputs
        
        logger.info('Performing membership, step by step')
        logger.debug('Query: %s' % inputs)
        
        config = Config()
        seed = Seed(CompositeData())

        executor = SymbolicExecutor(config, seed)
        executor.load(self.project)
                
        #executor.pstate.input = inputs[0]
        executor.pstate.monitoring_position = 0
        #executor.pstate.probing_pending = False
        #executor.pstate.done_probing = False
        #executor.pstate.probing_results = []
        #executor.pstate.probing_result_type = None
        #executor.pstate.probing_symbolic_var = None
        #executor.pstate.probed_symbol = None
        
        self.set_membership_hooks()
        logger.info('finnished setting hooks')
        executor.run()
        
        #monitoring success
        if self.monitoring_result:
            pass

    
    
