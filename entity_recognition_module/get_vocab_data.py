from tqdm import tqdm
word_set = set()
label_list = ["dev","test","train"]

for label in label_list:
    with open(f"data/Enterprise/{label}.txt", "r", encoding="UTF-8") as fp:
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
    with open("data/Enterprise/vocab.txt", "a+", encoding="UTF-8") as fp:
        fp.write(word+"\n")
