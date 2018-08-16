len_2 = ['樂']
def qulify_sentence(sentence):
    w = []
    tmp = []
    for i in range(len(sentence)):
        if i == 0:
            w.append(sentence[i])
            tmp.append(sentence[i])
        else:
            if sentence[i] in len_2:
                limit_num = 1
            else:
                limit_num = 2
            if len(tmp)>limit_num:
                if sentence[i] == tmp[-1]: continue
                else:
                    tmp = [sentence[i]]
                    w.append(sentence[i])
            else:
                if sentence[i] == tmp[-1]:
                    tmp.append(sentence[i])
                else:
                    tmp = [sentence[i]]
                w.append(sentence[i])
    return ('').join(w)
