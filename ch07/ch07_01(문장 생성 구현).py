import sys
sys.path.append('..')
import numpy as np
from common.functions import softmax
from ch06.rnnlm import Rnnlm
from ch06.better_rnnlm import BetterRnnlm

class RnnlmGen(Rnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100):
    # sart_id : 최초로 주는 단어의 ID, sample_size : sampling 하는 단어의 수, skip_ids : 단어 ID의 리스트
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x) # 각 단어의 점수 출력
            p = softmax(score.flatten()) # Softmax로 정규화

            sampled = np.random.choice(len(p), size=1, p=p)
            if(skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))
    
        return word_ids