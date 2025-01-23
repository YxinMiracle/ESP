# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import TextIteratorStreamer
# from threading import Thread
#
# model_name = "/home/cyx/.cache/huggingface/hub/models--Qwen2-72B-Instruct"
# device = "cuda"  # the device to load the model onto
#
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# system_prompt = """
# As a threat intelligence expert, you are about to be assigned a task which will require your meticulous attention. I have conducted a detailed analysis and processing of a threat intelligence report. Please assist in completing the task by following these steps:
#
# Step 1: Entity Extraction and Relationship Mapping Entity Extraction: Utilizing the STIX 2.1 standard, I have extracted various threat entities from the intelligence data. Additionally, I have mapped the relationships between these entities, based on the Relationship Object (SRO) attributes defined in STIX 2.1, to construct a graph showing how these entities are interconnected. I will provide this relationship graph in the format of a list of triplets as follows:
# relation_graph = [
#     {
#         "subject": {"name": "entity name", "type": "entity type (as defined by STIX 2.1)"},
#         "relation": "relationship type (e.g., 'uses', 'targets')",
#         "object": {"name": "entity name", "type": "entity type (as defined by STIX 2.1)"}
#     },
#     ...
# ]
# You can use this triplet data to gain a deeper understanding of the main contents described in the threat intelligence.
#
# Step 2: Provision of IOC and Threat Indicator Data I will provide Indicators of Compromise (IOC) and threat indicator data related to the intelligence, primarily in the following five categories defined by STIX 2.1:
# file-hash: File hash values, used to uniquely identify specific files and commonly used to detect known malicious files.
# malware: Information about malicious software, including its name, type, version, and characteristics.
# file-name: Relevant filenames, which could be malicious or infected.
# windows-registry-key: Windows registry keys, potentially created, modified, or exploited by malware.
# process: Information about suspicious or malicious system processes.
# The provided IOC data format is as follows:
# ioc_data = {
#     "malware": ["xxx"],
#     "file-name": ["xxx"],
#     "windows-registry-key": ["xxx"],
#     "process": ["xxx"]
# }
#
# Step 3: Generation of Yara Rules. Based on the data provided in Step 1 for understanding the intelligence, you will use the IOC data from Step 2, selecting appropriate content to generate corresponding Yara rules. Please note that not all the data provided may be useful; you need to analyze it based on your own experience to detect related malicious activities. Note, only use the IOC data provided in Step 2. The final rule output format should be as follows:
# {
#     "ruleName": "xxx",
#     "description": "xxx",
#     "variableList": [
#         {"variableName": "$xxx", "variableValue": "xxx"},
#         {"variableName": "$xxx", "variableValue": "xxx.exe"}
#         ...
#     ],
#     "condition": "xxx"
# }
# For the field ruleName, it refers to the detection theme related to this intelligence; give this rule a name.
# For the field description, it is a brief description of this intelligence.
# For the field variableList, it sets up the variable part of this Yara rule, where each variableName starts with a ‘$’ symbol and each variableName must be unique. The variableValue corresponds to the entities provided in Step 2. Please note, you need to consider the actual situation, and not necessarily use every entity provided!! You need to think for yourself.
# For the field condition, In YARA rules, the condition field allows users to precisely define the rule's trigger conditions using a variety of keywords and operators. These include Boolean logic operators (such as and, or, not) for controlling basic logical expressions; quantifiers (such as all of them, any of them, N of them) to specify the required number of patterns that must match; numerical conditions, which allow for comparisons of file sizes, the number of string matches, etc. (using <, <=, >, >=, ==, !=); location and range operators (such as at, in) to specify the exact location or range where a pattern must occur; and regular expressions, which are used to define complex matching conditions directly within the conditions. The combination of these features makes YARA a very powerful and flexible tool in areas such as malware analysis and digital forensics.
#
# Please provide the data in the specified format and refrain from including any other content.
#
# ***ONLY RESPOND IN this JSON FORMAT. No non-JSON text responses***
# { "ruleName": "", "description": ".", "variableList": [ {"variableName": "", "variableValue": ""}, {"variableName": "", "variableValue": ""}], "condition": "" }
# """
#
# prompt = """
# relation_graph =[
#     {
#         "subject": {
#             "name": "beacon,",
#             "type": "malware"
#         },
#         "relation": "downloads",
#         "object": {
#             "name": "iexplore.exe",
#             "type": "file-name"
#         }
#     },
#     {
#         "subject": {
#             "name": "Cobalt strike",
#             "type": "malware"
#         },
#         "relation": "downloads",
#         "object": {
#             "name": "iexplore.exe",
#             "type": "file-name"
#         }
#     },
#     {
#         "subject": {
#             "name": "Cobalt strike",
#             "type": "malware"
#         },
#         "relation": "uses",
#         "object": {
#             "name": "beacon",
#             "type": "malware"
#         }
#     },
#     {
#         "subject": {
#             "name": "Meterpreter",
#             "type": "malware"
#         },
#         "relation": "drops",
#         "object": {
#             "name": "MSBuild",
#             "type": "tool"
#         }
#     },
#     {
#         "subject": {
#             "name": "MSBuild",
#             "type": "tool"
#         },
#         "relation": "drops",
#         "object": {
#             "name": "Cobalt strike",
#             "type": "malware"
#         }
#     },
#     {
#         "subject": {
#             "name": "MSBuild",
#             "type": "tool"
#         },
#         "relation": "drops",
#         "object": {
#             "name": "zlib",
#             "type": "malware"
#         }
#     },
#     {
#         "subject": {
#             "name": "MSBuild",
#             "type": "tool"
#         },
#         "relation": "drops",
#         "object": {
#             "name": "beacon",
#             "type": "malware"
#         }
#     },
#     {
#         "subject": {
#             "name": "zlib",
#             "type": "malware"
#         },
#         "relation": "uses",
#         "object": {
#             "name": "base64",
#             "type": "tool"
#         }
#     },
#     {
#         "subject": {
#             "name": "MSBuild",
#             "type": "tool"
#         },
#         "relation": "drops",
#         "object": {
#             "name": "Meterpreter",
#             "type": "malware"
#         }
#     }
# ]
#
# ioc_data ={
#     "malware": [
#         "Meterpreter",
#         "zlib",
#         "Grunt",
#         "Cobalt strike",
#         "beacon",
#         "shellcode/beacon",
#         "beacon,",
#         "Meterpreter,",
#         "Cobalt Strike,"
#     ],
#     "file-name": [
#         "powershell.exe",
#         "csc.exe",
#         "cvtres.exe.",
#         "(iexplore.exe).",
#         "iexplore.exe",
#         "winword.exe",
#         "MSBuild.exe",
#         "a.bat",
#         "Mshta.exe",
#         "MSBuild.exe,",
#         "C:\\Windows\\Microsoft.Net\\Framework\\v4.0.30319\\Microsoft.Build.Tasks.v4.0.dll,"
#     ],
#     "malware_keylogger": [
#         "Mimikatz"
#     ],
#     "malware_remote-access-trojan": [
#         "Silent Trinity"
#     ]
# }
# """
#
# messages = [
#     {"role": "system",
#      "content": system_prompt},
#     {"role": "user", "content": prompt}
# ]
#
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )
#
# model_inputs = tokenizer([text], return_tensors="pt").to(device)
#
#
# streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
#
# generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512)
# thread = Thread(target=model.generate, kwargs=generation_kwargs)
# thread.start()
#
# generated_text = ""
# for new_text in streamer:
#     generated_text += new_text
#     print(new_text)
#
# print(generated_text)