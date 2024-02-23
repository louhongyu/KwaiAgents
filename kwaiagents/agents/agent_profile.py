import json


#1.定义私有变量名：_profile_default_name_fn；只能在定义它的模板和内部使用
#当输入的是”zh“时，返回”ai助手
_profile_default_name_fn = lambda x: "AI助手" if x == "zh" else "AI Assitant"   
_profile_default_bio_fn = lambda x: "你能帮助人类解决他们的问题" if x == "zh" else "You can help people solve their problems"
_profile_default_instruct_pre_fn = lambda x: "你要遵循以下指令行动：\n" if x == "zh" else "You should follow these following instructions:\n"


#定义助理的简介类，包括名称、指令、简介和工具
class AgentProfile(object):
    #初始化被调用的类，接受名为input_dict的参数，参数默认是None
    def __init__(self, input_dict: dict = None):
        #设置一个内部属性，接收传递进来的input_dict参数具体值
        self.input_dict = input_dict
        #设置一个内部属性lang，获取input_dict中“lang”键下的值，如无则默认“en”
        self.lang = input_dict.get("lang", "en")
        #调用一个名为from_json的内部方法，并将input_dict参数传递
        self.from_json(input_dict)

    #定义一个内部函数，处理input_dict参数，提取和设置代理配置文件的属性
    def from_json(self, input_dict):
        #在from_json函数中，设置一个内部的名为name的属性，作用是从input_dict参数中获取“agent_name"这个键的值，如无则空。
        self.name = input_dict.get("agent_name", "")
        #如果name属性为空，执行下面的代码
        if not self.name:
            #调用之前定义的 _profile_default_name_fn 函数，传入语言属性 self.lang，来获取默认的名字，并设置给 name 属性
            self.name = _profile_default_name_fn(self.lang)
        #在from_json函数中，设置一个内部的名为bio的属性，作用是从input_dict参数中获取“agent_bio"这个键的值，如无则空。
        self.bio = input_dict.get("agent_bio", "")
        #如果input_dict中没有agent_bio键值对，则调用下面的代码
        if not self.bio:
            #调用之前定义好的_profile_default_bio_fn函数，传入语言属性self.lang，获取默认的bio设置
            self.bio = _profile_default_bio_fn(self.lang)
        #在from_json函数中定义一个最大迭代轮次的属性，从input_dict中获取max_iter_num键值，若无就用默认的5轮
        self.max_iter_num = int(input_dict.get("max_iter_num", 5))
        #在from_json函数中定义一个指导语的属性，从input_dict中获取agent_instructions"键值，若无则传空字符串。
        self.instructions = input_dict.get("agent_instructions", "")
        #如果instructions非空，执行下面的代码；
        if self.instructions:
            #调用_profile_default_instruct_pre_fn内部方法，传入self.lang属性值，获取默认的instructions；把默认的instructions和input_dict中字符串拼接
            self.instructions = _profile_default_instruct_pre_fn(self.lang) + self.instructions
        #设置一个局部变量 tool_names，从 input_dict 中获取 tool_names 的值，如果没有则设置为一个包含单个元素 "auto" 的列表
        tool_names = input_dict.get("tool_names", '["auto"]')
        #检查tool_names是否为字符串类型，如果是，执行下面这个代码
        if isinstance(tool_names, str):
            #将字符串“tool_names",解析为json；将json编码的字符串转换成python对象，将结果赋值给tools属性。？？？：为什么不能直接将字符串复制给tools属性？一定要经过json解析编码一遍吗？
            self.tools = json.loads(tool_names)
        else:
            #如果它不是字符串，而是上面定义的列表，那么就直接将tool_names的值赋值给tools属性
            self.tools = tool_names
            
    #在 AgentProfile类中定义了一个to_json_file的方法，方法中接收fname参数
    def to_json_file(self, fname):
        #使用 with 语句打开一个文件，文件名由 fname 指定，模式为写入 ("w"), 并将文件对象赋值给变量 f。with 语句确保文件在使用后正确关闭。
        with open(fname, "w") as f:
            #这一行使用 json.dump 方法将对象 self 的 __dict__ 属性（包含所有实例属性的字典）转换为JSON格式并写入到文件 f 中。这个字典推导式 {k:v for k, v in self.__dict__.items()} 用于创建一个新的字典，
            #其中不包含任何不需要写入JSON的属性。ensure_ascii=False 参数允许非ASCII字符在JSON文件中保持原样，indent=2 参数使得JSON输出格式化，每个级别缩进2个空格。
            json.dump({k:v for k, v in self.__dict__.items()},f, ensure_ascii=False, indent=2)

    #两个下划线，是类的特殊方法；类的特殊方法 __str__，它定义了当一个实例被转换为字符串时的行为（比如当你调用 print(instance_of_AgentProfile) 时）。
    def __str__(self):
        s = "============ Agent Profile ============\n"
        #遍历 self 的 __dict__ 属性中的所有键值对。__dict__ 属性包含了对象的所有属性。
        for key, val in self.__dict__.items():
            #检查当前的键是否是 "input_dict"，如果是，则执行下一行代码。
            if key == "input_dict":
                #continue 语句跳过当前循环的剩余部分，因此如果键是 "input_dict"，则不会在字符串 s 中添加该键值对。
                continue
            #使用格式化字符串，将当前键转换为大写并添加到 s 字符串中，键和值之间用制表符分隔，并在末尾添加一个换行符。
            s += f"· {key.upper()}:\t{val}\n"
        #返回构建好的字符串 s。
        return s
 #这两代码的作用是将 AgentProfile 类的实例的内容以两种方式输出：一种是保存为JSON文件，另一种是生成一个易于阅读的多行字符串表示，其中不包括 input_dict 属性
