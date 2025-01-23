# from simpletransformers.classification import MultiLabelClassificationModel, MultiLabelClassificationArgs
from ttp_model.MutiLabelModel import MutiLabelModel
from ttp_model.ttp_config import MODEL_NAME, TECHNIQUE_MODEL_PATH, TECHNIQUE, \
    TACTICS_TECHNIQUES_RELATIONSHIP_DF
import spacy
from ttp_model.ttp_nlp_utils import preprocess
import torch
import numpy as np

# 加载预训练的英语模型
nlp = spacy.load("en_core_web_sm")


class CtiTTpModel:
    def __init__(self):
        self.te_multi_model = self.load_model(MODEL_NAME, TECHNIQUE_MODEL_PATH)
        self.te2ta_dict = {}

    def load_model(self, model_name, model_path):
        """
        加载模型
        :param model_name: 模型的名称
        :param model_path: 训练好的模型路径
        :return:
        """
        model = MutiLabelModel(576, "TECHNIQUE")
        model.load_state_dict(torch.load(model_path))
        model.cuda()
        print(f"{model_path} 加载成功")
        return model

    def split_cti_content(self, content: str):
        """
        切分句子，后面用作ttp识别
        :param content:
        :return:
        """
        doc = nlp(content)
        sent_list = [sent.text for sent in doc.sents]
        return sent_list

    # 定义函数来通过值查找对应的 key
    def find_tactics_by_techniques(self, techniques):
        if techniques in self.te2ta_dict:
            return self.te2ta_dict[techniques]
        for key, series in TACTICS_TECHNIQUES_RELATIONSHIP_DF.items():
            if techniques in series.values:
                self.te2ta_dict[techniques] = key
                return key
        return None

    def get_cti_ttp_result(self, content: str):
        """
        得到cti抽取ttp的结果分为：
        1. 文章级别
        2. 句子级别
        :param content:
        :return:
        """
        sent_list = self.split_cti_content(content)
        clear_sent_list = [preprocess(sent) for sent in sent_list]
        evaluate_res_list = []
        # 获取技术的结果
        te_predictions = self.te_multi_model(clear_sent_list)
        evaluate_res_list.extend(te_predictions.data.cpu().numpy())
        evaluate_res_list = np.array(evaluate_res_list)
        evaluate_res_list_bi = (evaluate_res_list >= 0.5).astype(int)
        # 用作判断这是否已经添加过
        inserted_ttp = []
        # 存放文章级别的ttp和句子级别的ttp
        cti_te_res, sent_te_list = [], []

        for sent, te_prediction_list in zip(sent_list, evaluate_res_list_bi):
            sent_ttp_information = []
            for technique, p in zip(TECHNIQUE, te_prediction_list):
                if p != 1: continue
                single_data = {
                    "techniqueID": technique,
                    "teactic": self.find_tactics_by_techniques(technique)
                }
                sent_ttp_information.append(single_data)
                if technique not in inserted_ttp:
                    cti_te_res.append(single_data)
            sent_te_list.append({
                "sent": sent,
                "techniqueList": sent_ttp_information
            })

        return cti_te_res, sent_te_list

if __name__ == '__main__':
    c = CtiTTpModel()
    c.get_cti_ttp_result("""
    Threat actors with an array of motivations continue to seek opportunities to exploit the digital infrastructure that Mexicans rely on across all aspects of society . This joint blog brings together our collective understanding of the cyber threat landscape impacting Mexico , combining insights from Google 's Threat Analysis Group ( TAG ) and Mandiant 's frontline intelligence . Since 2020 , cyber espionage groups from more than 10 countries have targeted users in Mexico ; however , more than 77 % of government - backed phishing activity is concentrated among groups from the People 's Republic of China ( PRC ) , North Korea , and Russia . Figure 1 : Government - backed phishing activity targeting Mexico , January 2020 – August 2024The examples here highlight recent and historical examples where cyber espionage actors have targeted users and organizations in Mexico . This volume of PRC cyber espionage is similar to activity in other regions where Chinese government investment has been focused , such as countries within China 's Belt and Road Initiative . One of the emerging trends we are witnessing globally from North Korea is the insider threat posed by North Korean nationals gaining employment surreptitiously at corporations to conduct work in various IT roles . Since 2020 , Russian cyber actors have accounted for approximately one - fifth of government - backed phishing activity targeting Mexico . These capabilities have grown the demand for spyware technology , making way for a lucrative industry used to sell to governments and nefarious actors the ability to exploit vulnerabilities in consumer devices . Over the past several years , open sources have reported multiple cases involving the use of spyware to target many sectors of Mexican civil society , including journalists , activists , government officials , and their families in Mexico . TAG has previously highlighted the negative outcomes of commercial spyware tools , including the proliferation of sophisticated cyber threat capabilities to new operators and sponsors , the increasing rates of zero - day vulnerability discovery and exploitation , and harm to targets of these tools . Though the use of spyware typically only affects a small number of human targets at a time , its wider impact ripples across society by contributing to growing threats to free speech and the free press and the integrity of democratic processes worldwide . TAG continues to observe evidence of several commercial surveillance vendors operating in Mexico . Notably , we have observed a variety of operations , including ransomware and extortion , targeting of banking credentials , cryptomining , and threat actors offering compromised access and/or credentials for sale . TAG continues to detect and disrupt multiple financially motivated groups targeting users and organizations in Mexico . Of these groups , three of the top four most frequently observed groups in the past year have been initial access brokers for extortion groups . Mandiant observed evidence of threat actors using a variety of initial access vectors , including phishing , malvertising , infected USB drives , and password spray . Like other countries in the region , Mexico is affected by threat activity from actors primarily active in Latin America as well as operations with global reach . A significant amount of observed campaigns focus on stealing credentials for banking or other financial accounts , including use of banking trojans such as METAMORFO aka " Horabot , " BBtok , and JanelaRAT . Many threat actors in the Latin American underground appear to focus on simpler operations in which they can quickly and easily generate profits , such as payment card theft and fraud . Mandiant tracks multiple data leak sites ( DLSs ) dedicated to releasing victim data following ransomware and/or extortion incidents in which victims refuse to pay a ransom demand . From January 2023 to July 2024 , Mexico was surpassed only by Brazil as the Latin American and Caribbean country most affected by ransomware and extortion operations , based on counts of DLS listings , though the global distribution of extortion activity as indicated by DLS listings remains heavily skewed towards the U.S. , Canada , and Western Europe . The most frequently impacted sectors in Mexico include manufacturing , technology , financial services , and government . Throughout 2023 and into 2024 , Mandiant observed UNC4984 activity distributing either malicious browser extensions or the SIMPLELOADER downloader using multiple distribution vectors , including using email lures for malware distribution . The malicious websites leveraged in these campaigns often masquerade as tax- or financial - related Chilean or Mexican government websites , and the malicious browser extensions specifically target Mexican bank institutions . Figure 4 : UNC4984 website spoofing the Mexican Tax Administration Service ( SAT ) prompting users to download a malicious browser extensionAnother financially motivated group , tracked as UNC5176 , uses emails and malicious advertisement ( aka " malvertising " ) campaigns to compromise users from various countries , including Brazil , Mexico , Chile , and Spain . Mandiant observed multiple malicious email campaigns delivering the URSA ( aka Mispadu ) backdoor to Latin American organizations in multiple industries , including a December 2023 UNC5176 campaign spoofing Mexico 's state - owned electric utility , the Comisión Federal de Electricidad . In April 2024 , an UNC5176 phishing campaign distributed URSA to organizations primarily located in Latin America using malicious PDF attachments containing an embedded link to a ZIP archive . In some incidents , the ZIP archives were hosted and retrieved from legitimate file - hosting services such as S3 buckets , Azure , Github , and Dropbox . Chrome OS has built - in , proactive security measures to protect from ransomware attacks , and there have been no reported ransomware attacks ever on any Chrome OS device . Google security teams continuously monitor for new threat activity , and all identified websites and domains are added to Safe Browsing to protect users from further exploitation . We also deploy and constantly update Android detections to protect users ' devices and prevent malicious actors from publishing malware to the Google Play Store . We send targeted Gmail and Workspace users government - backed attacker alerts , notifying them of the activity and highly encouraging device updates and the use of Enhanced Safe Browsing for Chrome . Global cyber espionage actors from the PRC , North Korea , and Russia as well as multinational cyber criminals pose longstanding threats . We hope the analysis and research here helps to inform defenders in Mexico , providing fresh insights for collective defense . At Google , we are committed to supporting the safety and security of online users everywhere and will continue to take action to disrupt malicious activity to protect our users and enterprise customers and help make the internet safe for all . Posted inThreat Intelligence
    """)