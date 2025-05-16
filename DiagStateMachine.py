import json
import os
import random
from typing import Dict, List, Tuple, Union

class HierarchicalStateMachine:
    def __init__(self, current_group, current_state, config_folder: str):

        self.current_group = current_group
        self.current_state = current_state
        self.current_subgroup = None
        self.group_rules: Dict[str, Dict] = {}
        self.cross_rules: Dict[Tuple[str, str], Tuple] = {}
        self.subgroup_handlers: Dict[str, callable] = {} 
        self.current_time = "time0"  
        self._load_rules(config_folder)
        self._reset_state()

    def _reset_state(self):
        self.substate_queue: List[Tuple[str, str]] = []
        self.state_values: Dict[str, Dict[str, bool]] = {}
        self.state_history = [{
            "group": self.current_group,
            "state": self.current_state,
            "timestamp": 0
        }]

    def _load_rules(self, folder: str):
        ingroup_rule_path = os.path.join(folder, "ingroup_rules.json")
        if os.path.exists(ingroup_rule_path):
            with open(ingroup_rule_path, 'r', encoding='utf-8') as f:
                self.group_rules = json.load(f)
        else:
            raise FileNotFoundError(f"组规则文件 {ingroup_rule_path} 不存在")
      
    
        cross_rule_path = os.path.join(folder, "crossgroup_rules.json")
        if os.path.exists(cross_rule_path):
            with open(cross_rule_path, 'r') as f:
                cross_rules_raw = json.load(f)
                self.cross_rules = {
                    tuple(k.split(':')): tuple(v) 
                    for k, v in cross_rules_raw.items()
                }
        else:
            raise FileNotFoundError(f"跨组规则文件 {cross_rule_path} 不存在")
        
        self._register_subgroup_handlers()

    def _register_subgroup_handlers(self):
        self.subgroup_handlers = {
            "A00": self._handle_A00,
            "A01": self._handle_A01,
            "A02": self._handle_A02,
            "A03": self._handle_A03,
            "A04": self._handle_A04,
            "F00": self._handle_F00,
            "F01": self._handle_F01,
            "K00": self._handle_K00,
            "K01": self._handle_K01
        }

 
 
    def _handle_A00(self) -> Tuple[List[str], int]:
        required = ["A3", "A6", "A9", "A12", "A13", "A16", "A17"]
        return (required, 5)

    def _handle_A01(self) -> Tuple[List[str], int]:
        required = ["A98", "A111", "A114", "A117", "A118", "A121", "A122"]
        return (required, 5)

    def _handle_A02(self) -> Tuple[List[str], int]:
        required = ["A138", "A139", "A140", "A141", "A142", "A143", "A144", "A145", "A146", "A147"]
        return (required, 3)

    def _handle_A03(self) -> Tuple[List[str], int]:
        required = ["A255", "A256", "A257", "A258", "A259", "A260", "A261", "A262", "A263", "A264"]
        return (required, 3)

    def _handle_A04(self) -> Tuple[List[str], int]:
        required = ["A286", "A287", "A289", "A290", "A291", "A292", "A293", "A294", "A295", "A288"]
        return (required, 3)

    def _handle_F00(self) -> Tuple[List[str], int]:
        required = ["F144", "F145", "F146", "F147", "F148", "F149"]
        return (required, 3)

    def _handle_F01(self) -> Tuple[List[str], int]:
        required = ["F169", "F170", "F171", "F172", "F173", "F174"]
        return (required, 3)

    def _handle_K00(self) -> Tuple[List[str], int]:
        return (["K6", "K7", "K8", "K9", "K10", "K11", "K12", "K13","K14"], 5)

    def _handle_K01(self) -> Tuple[List[str], int]:
        return (["K16", "K17", "K18", "K19", "K20", "K21", "K22", "K23","K24"], 5)



    def get_next_state(self, response: Union[bool, str]) -> Dict:
        is_yes = bool(response)
        if self.substate_queue:
            return self._process_substate(is_yes)

        if cross_info := self._check_cross_rule():
            return cross_info
        return self._process_normal_state(is_yes)

    def _process_substate(self, is_yes: bool) -> Dict:
        """Handling sub-state group logic"""
        self._record_response(self.current_state, is_yes)
        current_state = self.current_state
        if current_state in self.group_rules[self.current_group][self.current_subgroup]:
            state_rule = self.group_rules[self.current_group][self.current_subgroup][current_state]
            next_key = "Y" if is_yes else "N"
            if state_rule[next_key] is not None:
                self.substate_queue.insert(1, (state_rule[next_key], 'time0')) 
        
        self.substate_queue.pop(0)        
        if not self.substate_queue:
            self._finalize_subgroup()
            return self._build_state_info()
        self.current_state = self.substate_queue[0][0]
        self.current_time = self.substate_queue[0][1]  
        self._record_history()  
        return self._build_state_info()

    def _process_normal_state(self, is_yes: bool) -> Dict:
        """Handling common state transitions"""
        state_rule = self.group_rules[self.current_group][self.current_state]
        subgroup_list = ["A00", "A01", "A02", "A03", "A04", "F00", "F01", "K00", "K01"]
        next_key = "Y" if is_yes else "N"
        if next_key in state_rule:
            next_state, next_time = state_rule[next_key] 
            if next_state in subgroup_list:
                self.current_subgroup = next_state
                self._init_substate_group(self.current_subgroup)                   
                return self._build_state_info()
            else:
                self.current_state = next_state                
                self.current_time = next_time                  
                self._record_history()
                return self._build_state_info()
            
        raise ValueError(f"Invalid state transition: {self.current_group}.{self.current_state} -> {next_key}")

    def _init_substate_group(self, group: str):
        """Initialize sub-state group"""
        time_options = {
            "A00": ["time1", ["time3", "time0","time0","time0","time3", "time0","time2"]],
            "A01": ["time1", ["time3", "time0","time0","time0","time3", "time0","time2"]],
            "A02": ["time4", ["time3","time5", "time0","time0","time3","time0", "time0"]],
            "A03": ["time4", ["time3","time5", "time0","time0","time3","time0", "time0"]],
            "A04": ["time12", ["time13","time0","time0","time3","time0", "time0","time3"]],
            "F00": ["time6", ["time0", "time7", "time8","time0","time0","time8","time0"]],
            "F01": ["time10", ["time0", "time9", "time3","time0","time0", "time3","time0"]],
            "K00": ["time11", ["time0", "time9", "time3","time0","time0", "time3","time0"]],
            "K01": ["time11", ["time0", "time9", "time3","time0","time0", "time3","time0"]]
        }
        print(f"subgroup{group}")
        first_time, other_times = time_options[group]
        handler = self.subgroup_handlers[self.current_subgroup]
        required_states, threshold = handler()
        random.shuffle(required_states)
        for i in range(len(required_states)):
            if i == 0:
                self.substate_queue.append((required_states[i], first_time))
            else:
                self.substate_queue.append((required_states[i], random.choice(other_times)))
        print(f"required_states {required_states} ")
        print(f"Sub-state queue {self.substate_queue} ")
        self.current_state = required_states[0] 
        self.current_time = first_time

        self.state_values.setdefault(self.current_subgroup, {}).clear()
        print(f"Sub-state group {self.current_subgroup} is initialized, current state: {self.current_state}, current time: {self.current_time}")
        self._record_history()


    def _finalize_subgroup(self):
        """Complete sub-state group processing"""
        y_count = sum(self.state_values[self.current_subgroup].values())
        subgroup = self.current_subgroup
        response = y_count >= self._get_threshold()
        self.current_state = subgroup
        self.current_subgroup = None 
        self._process_normal_state(response)
        


    def _get_threshold(self) -> int:
        """Get the current sub-state group threshold"""
        handler = self.subgroup_handlers[self.current_subgroup]
        return handler()[1]

    
    def _check_cross_rule(self) -> Union[Dict, None]:
        """Perform cross-group jumps and verify target validity"""
  
        cross_key = (self.current_group, self.current_state)  
        if cross_key in self.cross_rules:
           
            target_group, target_state, target_time = self.cross_rules[cross_key]
            if target_group not in self.group_rules:
                raise ValueError(f"目标组 {target_group} 未在规则中定义")

            if target_state not in self.group_rules[target_group]:
                raise ValueError(f"目标状态 {target_state} 未在组 {target_group} 中定义")
  
            self.current_group = target_group
            self.current_state = target_state
            self.current_time = target_time
            return self._build_state_info()
        return None

    def _record_response(self, state: str, value: bool):
        """Logging substatus responses"""
        self.state_values.setdefault(self.current_subgroup, {})[state] = value

    def _record_history(self):
        if self.current_subgroup is not None:
             self.state_history.append({
                 "group": self.current_group,
                 "subgroup": self.current_subgroup,
                 "state": self.current_state,
                 "timestamp": len(self.state_history)
             })
          
        else:
            self.state_history.append({
                "group": self.current_group,
                "state": self.current_state,
                "timestamp": len(self.state_history)
            })

    def _build_state_info(self) -> Dict:
        """Constructing a state information dictionary"""
        if self.current_subgroup is not None:
            return {
                "current_group": self.current_group,
                "current_subgroup": self.current_subgroup,
                "current_state": self.current_state,
                "current_time": self.current_time,  
                "substate_progress": f"{len(self.substate_queue)-1} remaining",
                "history": self.state_history[-8:]
                }
        else:
            return {
                "current_group": self.current_group,
                "current_state": self.current_state,
                "current_time": self.current_time,  
                "substate_progress": f"{len(self.substate_queue)} remaining",
                "history": self.state_history[-8:]
            }

if __name__ == "__main__":
    machine = HierarchicalStateMachine("A.1","A1","prompts/diagstatemachine")
    print(machine.current_state)
    print("\n")
    print(machine.get_next_state(False)) 
    print(machine.get_next_state(True))
