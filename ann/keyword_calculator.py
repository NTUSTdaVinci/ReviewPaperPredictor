import numpy as np
import pandas as pd
import re


class KeywordCalculator:

    def __init__(self, ti, ab, wb):
        self.title_article = ti
        self.abstract_article = ab
        self.wordBank = wb.values.tolist()
        self.wordBank_list = np.append(wb['keyword'], wb['keyword'])
        self.input_data = []
        self.saveDir = []

    def get_result(self):
        input_counter = 0

        print("Calculating keyword count in training data ...")
        for ti_sentence, ab_sentence in zip(self.title_article, self.abstract_article):
            ti_list = []
            ab_list = []
            # calculate the number of each keyword in title and abstract
            # save the result in ti_list and ab_list
            for word_list in self.wordBank:
                word_counter_ti = 0
                word_counter_ab = 0
                print(word_list)
                for word in word_list:
                    if pd.isna(word):
                        break
                    word_counter_ti += len(re.findall(word.lower(), str(ti_sentence).lower(), re.IGNORECASE))
                    word_counter_ab += len(re.findall(word.lower(), str(ab_sentence).lower(), re.IGNORECASE))
                ti_list.append(word_counter_ti)
                ab_list.append(word_counter_ab)

            sentence_list = ti_list + ab_list
            ti_list.clear()
            ab_list.clear()
            self.input_data.append(sentence_list)
            input_counter += 1

        return np.array(self.input_data)

    def save_statistics(self, dir):
        self.saveDir = dir
        df_word = pd.DataFrame(self.input_data, columns=self.wordBank_list)
        df_word.to_excel(self.saveDir + "/keywords statistics.xlsx")