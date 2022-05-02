import sys
sys.path.append('..')
from common.util import preprocess, create_co_matrix, ppmi
import numpy as np
import matplotlib.pyplot as plt

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

# SVD
U, S, V = np.linalg.svd(W)

print(C[0]) # 동시 발생 행렬
print(W[0]) # PPMI 행렬
print(U[0]) # SVD
print(U[0, :2]) # 2차원 벡터로 줄인다 -> 처음의 두 원소를 꺼낸다.

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

plt.scatter(U[:,0], U[:,1], alpha=0.5)
plt.show()