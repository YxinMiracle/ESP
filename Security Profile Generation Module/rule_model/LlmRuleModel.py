import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,TextIteratorStreamer
from threading import Thread

from rule_model.llm_model_config import LLM_MODEL_PATH


class LLM_RULE_MODEL:



    SYSTEM_YARA_PROMPT = """
    As a threat intelligence expert, you are about to be assigned a task which will require your meticulous attention. I have conducted a detailed analysis and processing of a threat intelligence report. Please assist in completing the task by following these steps:

    Step 1: Entity Extraction and Relationship Mapping Entity Extraction: Utilizing the STIX 2.1 standard, I have extracted various threat entities from the intelligence data. Additionally, I have mapped the relationships between these entities, based on the Relationship Object (SRO) attributes defined in STIX 2.1, to construct a graph showing how these entities are interconnected. I will provide this relationship graph in the format of a list of triplets as follows:
    relation_graph = [
        {
            "subject": {"name": "entity name", "type": "entity type (as defined by STIX 2.1)"},
            "relation": "relationship type (e.g., 'uses', 'targets')",
            "object": {"name": "entity name", "type": "entity type (as defined by STIX 2.1)"}
        },
        ...
    ]
    You can use this triplet data to gain a deeper understanding of the main contents described in the threat intelligence.

    Step 2: Provision of IOC and Threat Indicator Data I will provide Indicators of Compromise (IOC) and threat indicator data related to the intelligence, primarily in the following five categories defined by STIX 2.1:
    file-hash: File hash values, used to uniquely identify specific files and commonly used to detect known malicious files.
    malware: Information about malicious software, including its name, type, version, and characteristics.
    file-name: Relevant filenames, which could be malicious or infected.
    windows-registry-key: Windows registry keys, potentially created, modified, or exploited by malware.
    process: Information about suspicious or malicious system processes.
    The provided IOC data format is as follows:
    ioc_data = {
        "malware": ["xxx"],
        "file-name": ["xxx"],
        "windows-registry-key": ["xxx"],
        "process": ["xxx"]
    }

    Step 3: Generation of Yara Rules. Based on the data provided in Step 1 for understanding the intelligence, you will use the IOC data from Step 2, selecting appropriate content to generate corresponding Yara rules. Please note that not all the data provided may be useful; you need to analyze it based on your own experience to detect related malicious activities. Note, only use the IOC data provided in Step 2. The final rule output format should be as follows:
    {
        "ruleName": "xxx", 
        "description": "xxx", 
        "variableList": [
            {"variableName": "$xxx", "variableValue": "xxx"},
            {"variableName": "$xxx", "variableValue": "xxx.exe"}
            ...
        ],
        "condition": "xxx" 
    }
    For the field ruleName, it refers to the detection theme related to this intelligence; give this rule a name. 
    For the field description, it is a brief description of this intelligence. 
    For the field variableList, it sets up the variable part of this Yara rule, where each variableName starts with a ‘$’ symbol and each variableName must be unique. The variableValue corresponds to the entities provided in Step 2. Please note, you need to consider the actual situation, and not necessarily use every entity provided!! You need to think for yourself.
    For the field condition, In YARA rules, the condition field allows users to precisely define the rule's trigger conditions using a variety of keywords and operators. These include Boolean logic operators (such as and, or, not) for controlling basic logical expressions; quantifiers (such as all of them, any of them, N of them) to specify the required number of patterns that must match; numerical conditions, which allow for comparisons of file sizes, the number of string matches, etc. (using <, <=, >, >=, ==, !=); location and range operators (such as at, in) to specify the exact location or range where a pattern must occur; and regular expressions, which are used to define complex matching conditions directly within the conditions. The combination of these features makes YARA a very powerful and flexible tool in areas such as malware analysis and digital forensics.

    Please provide the data in the specified format and refrain from including any other content.

    ***ONLY RESPOND IN this JSON FORMAT. No non-JSON text responses***
    { "ruleName": "", "description": ".", "variableList": [ {"variableName": "", "variableValue": ""}, {"variableName": "", "variableValue": ""}], "condition": "" }
    """

    YARA_QUESTION_TEMPLATE = """
        relation_graph = {relation_graph}
        ioc_data = {ioc_data}
    """

    SYSTEM_SNORT_PROMPT = """
    As a large model specializing in security, your task is to take the intelligence data I provide and return Snort rule data so that I can directly import it into the Snort intrusion detection system. The input for this task is primarily divided into two parts. One is domain-name, which contains malicious domain names extracted from intelligence. The second is ipv4-addr, which includes malicious IP addresses extracted from intelligence. I will input these to you in JSON format, like:
    {
        "domain-name": [
    
        ],
        "ipv4-addr": [
    
        ]
    }
    However, for domain names, the intelligence sometimes uses the [.] symbol to separate the parts of the domain to protect the information, such as in spl[.]noip[.]me. Before you write the Snort rules, you need to rewrite and deduplicate these domain names. For example:
    "domain-name": [
        "dennyhacker[.]no-ip.org",
        "spl[.]noip[.]me.",
        "spl[.]noip[.]me"
    ]
    After rewriting and deduplication, it will become [dennyhacker.no-ip.org,spl.noip.me].
    
    Here is your output format: 
    ***Only respond in this format:***
    
    {
        "ruleName": "",
        "description":"",
        "snortRuleList":[
            {
                "protocol":"The protocol used (e.g., TCP, UDP, IP)", 
                "destination": "The destination IP address or domain being connected to",
                "msg": "The alert message for this rule",
                "sid": "sid number",
                "rev": "revision number"
            },
        ]
    } 
    ONLY RESPOND IN JSON. No non-JSON text responses. Work through this step-by-step to ensure accuracy in the evaluation.
    """

    ABSTRACT_SYSTEM_PROMPT = """
    你作为威胁情报的专家，我将输入一篇英语威胁情报，你将对他进行结构化分析，输出应包括以下部分，具体内容依据情报类型选择适当的细节进行展示：
    
    概述层：
    威胁摘要：提供关于恶意软件或恶意流量的基本信息，如传播速度、影响程度、主要功能等。
    目标描述：描述攻击的主要目标，包括受影响的行业、系统类型、地理位置等信息。
    
    战术层：
    攻击目标：明确攻击者试图达到的目标，如数据窃取、系统破坏、勒索等。
    攻击手段：描述攻击者使用的具体技术或方法，例如钓鱼攻击、利用漏洞、社交工程等。
    攻击阶段：详细说明攻击的各个阶段，从初始接触到最终执行，包括攻击的进展和变化。
    
    技术层：
    恶意软件类型：如果是恶意软件，说明其类型（如病毒、蠕虫、木马等）及主要技术特性。
    流量特征：如果是恶意流量，分析其特征，如使用的协议、端口号、流量模式等。
    攻击工具和方法：列出用于执行攻击的工具和技术，包括任何已知的软件工具、代码库或利用工具。
    
    行为层：
    入侵行为：详细描述恶意行为的具体操作，如信息窃取的方式、持久化机制、与控制服务器的交互等。
    生命周期：分析恶意软件或流量的活动周期，从感染、活动到可能的清理或自我删除的过程。
    
    ------------------------------------------------------------------
    输出建议：
    确保信息准确、详尽，每一部分应清晰标识并归纳关键信息。
    根据情报的详细内容灵活调整每个部分的细节，确保全面覆盖所有相关信息。
    采用客观和技术性的语言描述情报，避免主观或模糊的表述。
    我输入的会是英语情报，你输出中文的摘要即可。
    ***注意，输出markdown格式的数据。***
    """

    def __init__(self):
        self.model_name = LLM_MODEL_PATH
        self.llm_model, self.llm_tokenizer = self._init_llm_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init_llm_model(self):
        """ 初始化本地大模型 """
        llm_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        llm_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return llm_model, llm_tokenizer

    def get_llm_answer(self, prompt: str, question: str) -> str:
        """ 获取大模型答案 """
        messages = [
            {"role": "system",
             "content": prompt},
            {"role": "user", "content": question}
        ]
        text = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.llm_model.generate(
            **model_inputs,
            max_new_tokens=4096
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def get_ans_streamer(self, prompt: str, question: str):
        """获取大模型的流式输出"""
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]

        text = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.llm_tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512)
        thread = Thread(target=self.llm_model.generate, kwargs=generation_kwargs)
        thread.start()
        for new_text in streamer:
            yield new_text