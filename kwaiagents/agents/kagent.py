from collections import deque
import json
import logging
import re
import sys
import time
import traceback
from typing import Dict, List
import uuid
from datetime import datetime
from lunar_python import Lunar, Solar
from transformers import AutoTokenizer

from kwaiagents.tools import ALL_NO_TOOLS, ALL_TOOLS, FinishTool, NoTool
from kwaiagents.llms import create_chat_completion
from kwaiagents.agents.prompts import make_planning_prompt
from kwaiagents.agents.prompts import make_no_task_conclusion_prompt, make_task_conclusion_prompt
from kwaiagents.utils.chain_logger import *
from kwaiagents.utils.json_fix_general import find_json_dict, correct_json
from kwaiagents.utils.date_utils import get_current_time_and_date


class SingleTaskListStorage:
    def __init__(self):
        #这一行创建了一个名为tasks的实例变量，它被初始化为一个空的双向队列（deque）。
        #双向队列是一种数据结构，可以在两端高效地进行插入和删除操作。
        self.tasks = deque([])
        #创建一个taskid的实例变量，生成每一个唯一的taskid
        self.task_id_counter = 0

    #将传入的任务（一个字典）添加到任务列表
    def append(self, task: Dict):
        self.tasks.append(task)

    #将当前任务列表self.tasks替换为一个新的双向队列，该队列初始化为传入的tasks列表
    def replace(self, tasks: List[Dict]):
        self.tasks = deque(tasks)
        
    #使用了双向队列（deque）的popleft()方法，从任务列表的左侧移除并返回第一个任务。
    def popleft(self):
        return self.tasks.popleft()

    #如果任务列表self.tasks非空，则返回False，否则返回True。它通过检查任务列表是否存在元素来确定列表是否为空。
    def is_empty(self):
        return False if self.tasks else True

    #生成下一个可用的任务ID
    def next_task_id(self):
        self.task_id_counter += 1
        return self.task_id_counter

    #返回任务列表中所有任务的名称
    def get_task_names(self):
        #列表推导式
        return [t["task_name"] for t in self.tasks]

    #返回任务列表，是任务列表的副本
    def get_tasks(self):
        return list(self.tasks)

    #将任务列表清空，重置为一个空的双向队列，并重置任务计数器为0
    def clear(self):
        del self.tasks
        self.tasks = deque([])
        self.task_id_counter = 0

#构造函数完成了对代理系统的各种属性的初始化，包括配置、会话 ID、分词器、日志记录器、内存和工具
class KAgentSysLite(object):
    def __init__(self, cfg, session_id=None, agent_profile=None, tools=None, lang="en"):
        self.cfg = cfg
        self.agent_profile = agent_profile
        self.lang = lang
        self.max_task_num = agent_profile.max_iter_num
        #如果提供了会话 ID，则将其赋值给 self.session_id 属性；
        #否则，使用 uuid.uuid1() 方法生成一个新的 UUID，并将其作为会话 ID。
        self.session_id = session_id if session_id else str(uuid.uuid1())
        #初始化代理系统的分词器，并将其赋值给 self.tokenizer 属性。
        self.tokenizer = self.initialize_tokenizer(self.cfg.fast_llm_model)
        #初始化代理系统的日志记录器。
        self.initialize_logger()
        #初始化代理系统的内存。
        self.initialize_memory()
        #初始化代理系统的工具，传入 tools 参数。
        self.tool_retrival(tools)

    #这一行代码创建了一个名为 chain_logger 的新的日志记录器对象。这个日志记录器基于 ChainMessageLogger 类，
    #它接受一个参数 output_streams，其中包含一个输出流列表。在这里，我们将输出流设置为 sys.stdout，
    #表示日志信息将被输出到标准输出流。另外，我们也传入了 lang 参数，指定日志记录器的语言设置。
    def initialize_logger(self):
        self.chain_logger = ChainMessageLogger(output_streams=[sys.stdout], lang=self.lang)
        #将新创建的日志记录器对象 chain_logger 设置为代理系统配置对象 cfg 的链式记录器。
        #这样做可以确保代理系统中的其他组件使用相同的日志记录器进行记录，从而集中管理日志信息。
        self.cfg.set_chain_logger(self.chain_logger)

    def initialize_memory(self):
        pass

    #使用 Hugging Face 的 AutoTokenizer.from_pretrained() 方法初始化分词器。
    def initialize_tokenizer(self, llm_name):
        if "baichuan" in llm_name:
            model_name = "kwaikeg/kagentlms_baichuan2_13b_mat"
        elif "qwen" in llm_name:
            model_name = "kwaikeg/kagentlms_qwen_7b_mat"
        else:
            model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False, #禁用快速分词器，以便获得更准确的结果。
            padding_side='left', #指定在序列填充时在序列的左侧添加填充标记。
            trust_remote_code=True #允许信任远程代码，用于从 Hugging Face 模型中加载分词器。
        )
        return tokenizer

    #初始化代理系统所需的工具
    def tool_retrival(self, tools):
        if tools:
            self.tools = [tool_cls(cfg=self.cfg) for tool_cls in tools]
        else:
            if "notool" in self.agent_profile.tools:
                self.tools = list()
            else:
                all_tools = [tool_cls(cfg=self.cfg) for tool_cls in ALL_TOOLS]

                if "auto" in self.agent_profile.tools:
                    used_tools = [tool_cls(cfg=self.cfg) for tool_cls in ALL_TOOLS]
                else:
                    used_tools = list()
                    for tool in all_tools:
                        if tool.zh_name in self.agent_profile.tools or tool.name in self.agent_profile.tools:
                            used_tools.append(tool)
                used_tools += [tool_cls(cfg=self.cfg) for tool_cls in ALL_NO_TOOLS]
            
            self.tools = used_tools
        #创建一个字典，将工具名称映射到对应的工具实例。
        self.name2tools = {t.name: t for t in self.tools}

    def memory_retrival(self, 
        goal: str, 
        conversation_history: List[List], 
        complete_task_list: List[Dict]):

        memory = ""
        if conversation_history:
            memory += f"* Conversation History:\n"
            #遍历最近的三个对话历史记录。
            for tmp in conversation_history[-3:]:
                memory += f"User: {tmp['query']}\nAssistant:{tmp['answer']}\n"

        if complete_task_list:
            #如果存在已完成任务列表，将其转换为 JSON 字符串形式，并存储在变量 complete_task_str 中。
            #ensure_ascii=False 参数确保非 ASCII 字符也被正确编码，indent=4 参数用于美化输出，以便易于阅读。
            complete_task_str = json.dumps(complete_task_list, ensure_ascii=False, indent=4)
            memory += f"* Complete tasks: {complete_task_str}\n"
        return memory

    def task_plan(self, goal, memory):
        prompt = make_planning_prompt(self.agent_profile, goal, self.tools, memory, self.cfg.max_tokens_num, self.tokenizer, lang=self.lang)
        # print(f'\n************** TASK PLAN AGENT PROMPT *************')
        # print(prompt)
        try:
            response, _ = create_chat_completion(
            query=prompt, llm_model_name=self.cfg.smart_llm_model)
            self.chain_logger.put_prompt_response(
                prompt=prompt, 
                response=response, 
                session_id=self.session_id, 
                mtype="auto_task_create",
                llm_name=self.cfg.smart_llm_model)
            response = correct_json(find_json_dict(response))
            task = json.loads(response)
            new_tasks = [task]
        except:
            print(traceback.format_exc())
            print("+" + response)
            self.chain_logger.put("fail", logging_think_fail_msg(self.lang))
            new_tasks = list()
        
        return new_tasks

    def tool_use(self, command) -> str:
        try:
            command_name = command.get("name", "")
            if command_name == "search":
                command_name = "web_search"
            args_text = ",".join([f'{key}={val}' for key, val in command["args"].items()])
            execute_str = f'{command_name}({args_text})'.replace("wikipedia(", "kuaipedia(")
            self.chain_logger.put("execute", execute_str)
            if not command_name:
                raise RuntimeError("{} has no tool name".format(command))
            if command_name not in self.name2tools:
                raise RuntimeError("has no tool named {}".format(command_name))
            tool = self.name2tools[command_name]

            tool_output = tool(**command["args"])
            self.chain_logger.put("observation", tool_output.answer_md)

            for prompt, response in tool_output.prompt_responses:
                self.chain_logger.put_prompt_response(
                    prompt=prompt,
                    response=response,
                    session_id=self.session_id,
                    mtype=f"auto_command_{command_name}",
                    llm_name=self.cfg.fast_llm_model
                )
            return tool_output.answer
        except KeyboardInterrupt:
            exit()
        #它捕获任何类型的异常并处理它们。
        #它将异常的堆栈跟踪信息打印出来，记录执行失败的日志消息，并返回一个空字符串作为异常处理的结果。
        except:
            print(traceback.format_exc())
            self.chain_logger.put("observation", logging_execute_fail_msg(self.lang))
            return ""
            
    #生成对话的结论，它根据对话的目标、记忆内容以及是否有计划任务来生成相应的结论。
    def conclusion(self, 
        goal: str, 
        memory,
        conversation_history: List[List],
        no_task_planned: bool = False
        ):

        if no_task_planned:
            prompt = make_no_task_conclusion_prompt(goal, conversation_history)
        else:
            prompt = make_task_conclusion_prompt(self.agent_profile, goal, memory, self.cfg.max_tokens_num, self.tokenizer, lang=self.lang)
        # print(f'\n************** CONCLUSION AGENT PROMPT *************')
        # print(prompt)

        response, _ = create_chat_completion(
            query=prompt, 
            chat_id="kwaiagents_answer_" + self.session_id, 
            llm_model_name=self.cfg.smart_llm_model)

        # print(response)

        self.chain_logger.put_prompt_response(
            prompt=prompt, 
            response=response, 
            session_id=self.session_id, 
            mtype="auto_conclusion",
            llm_name=self.cfg.smart_llm_model)
        return response

    #用于检查任务是否已完成
    def check_task_complete(self, task, iter_id):
        command_name = task["command"]["name"]
        if not task or ("task_name" not in task) or ("command" not in task) \
            or ("args" not in task["command"]) or ("name" not in task["command"]):
            self.chain_logger.put("finish", str(task.get("task_name", "")))
            return True
        elif command_name == FinishTool.name:
            self.chain_logger.put("finish", str(task["command"]["args"].get("reason", "")))
            return True
        elif command_name == NoTool.name:
            if iter_id == 1:
                self.chain_logger.put("finish", logging_do_not_need_use_tool_msg(self.lang))
            else:
                self.chain_logger.put("finish", logging_do_not_need_use_tool_anymore_msg(self.lang))
            return True
        elif command_name not in self.name2tools:
            self.chain_logger.put("finish", logging_do_not_need_use_tool_msg(self.lang))
            return True
        else:
            return False

    #这个方法是代理系统中的核心对话逻辑，包含了任务规划、执行和结论等过程。
    def chat(self, query, history=list(), initial_task_name=None, *args, **kwargs):
        goal = query

        if not self.tools:
            no_task_planned = True
        else:
            tasks_storage = SingleTaskListStorage()
            tasks_storage.clear()

            start = True
            loop = True
            iter_id = 0
            complete_task_list = list()
            no_task_planned = False
            #循环执行任务规划、执行和结论的过程：
            while loop:
                iter_id += 1
                if start or not tasks_storage.is_empty():
                    start = False
                    if not tasks_storage.is_empty():
                        task = tasks_storage.popleft()
                        
                        if (self.check_task_complete(task, iter_id,)):
                            if iter_id <= 2:
                                no_task_planned = True
                            break

                        self.chain_logger.put("thought", task.get("task_name", ""))

                        result = self.tool_use(task["command"])

                        task["result"] = result
                        complete_task_list.append(task)

                    if iter_id > self.agent_profile.max_iter_num:
                        self.chain_logger.put("finish", logging_stop_thinking_msg(self.lang))
                        break
                    self.chain_logger.put("thinking")
                    memory = self.memory_retrival(goal, history, complete_task_list)
                    new_tasks = self.task_plan(goal, memory)

                    for new_task in new_tasks:
                        new_task.update({"task_id": tasks_storage.next_task_id()})
                        tasks_storage.append(new_task)
                else:
                    loop = False
                    self.chain_logger.put("finish", logging_finish_task_msg(self.lang))

        memory = self.memory_retrival(goal, history, complete_task_list)
        self.chain_logger.put("conclusion", "")

        conclusion = self.conclusion(
            goal, 
            memory=memory,
            conversation_history=history,
            no_task_planned=no_task_planned)
        self.chain_logger.put("chain_end", "")

        new_history = history[:] + [{"query": query, "answer": conclusion}]
        #返回包含响应、更新后的对话历史记录、链式日志消息列表和字符串等信息的字典。
        return {
            "response": conclusion,
            "history": new_history,
            "chain_msg": self.chain_logger.chain_msgs,
            "chain_msg_str": self.chain_logger.chain_msgs_str,
            "more_info": {},
        }
