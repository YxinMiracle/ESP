# 开源数据中缺少了vocab数据，现在需要进行整合
from tqdm import tqdm
word_set = set()
label_list = ["dev","test","train"]

for label in label_list:
    with open(f"ner_data/Enterprise/{label}.txt", "r", encoding="UTF-8") as fp:
        line_data = fp.readlines()
        for line in line_data:
            line = line.replace("\n","")
            if line != "":
                try:
                    word, label = line.split(" ")
                    word_set.add(word)
                except:
                    pass

for word in tqdm(word_set):
    with open("ner_data/Enterprise/vocab.txt", "a+", encoding="UTF-8") as fp:
        fp.write(word+"\n")

"""

{'alias-of': 170,
 'attack-pattern': 23,
 'attributed-to': 20,
 'authored-by': 13,
 'beacons-to': 2,
 'campaign': 494,
 'communicates-with': 19,
 'compromises': 4,
 'consists-of': 18,
 'delivers': 28,
 'directory': 35,
 'domain-name': 879,
 'downloads': 21,
 'drops': 9,
 'email-addr': 80,
 'exploits': 50,
 'file-hash': 402,
 'file-name': 2244,
 'has': 5,
 'hashes-to': 18,
 'hosts': 24,
 'http-request-ext': 4,
 'identity': 2779,
 'identity_victim': 861,
 'infrastructure': 1204,
 'infrastructure_attack': 11,
 'infrastructure_command-and-control': 4,
 'infrastructure_hosting-malware': 64,
 'infrastructure_victim': 351,
 'intrusion-set': 1097,
 'ipv4-addr': 458,
 'ipv6-addr': 2,
 'located-at': 22,
 'location': 1376,
 'malware': 7576,
 'malware_bot': 80,
 'malware_ddos': 17,
 'malware_exploit-kit': 151,
 'malware_keylogger': 105,
 'malware_ransomware': 968,
 'malware_remote-access-trojan': 1373,
 'malware_screen-capture': 1,
 'malware_virus': 100,
 'malware_webshell': 161,
 'malware_worm': 35,
 'mutex': 7,
 'owns': 14,
 'process': 85,
 'resolves-to': 7,
 'software': 1442,
 'targets': 233,
 'threat-actor': 2511,
 'tool': 2773,
 'url': 221,
 'user-account': 2,
 'uses': 491,
 'variant-of': 81,
 'vulnerability': 470,
 'windows-registry-key': 52}

{'Area': 2530,
 'Exp': 1331,
 'Features': 1025,
 'HackOrg': 4210,
 'Idus': 1579,
 'OffAct': 1661,
 'Org': 1360,
 'Purp': 918,
 'SamFile': 1660,
 'SecTeam': 1327,
 'Time': 1675,
 'Tool': 3081,
 'Way': 1037}

{'Action': 2838, 'Entity': 6094, 'Modifier': 1786}

"""
