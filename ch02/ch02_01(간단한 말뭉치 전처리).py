import numpy as np

text = 'You say goodbye and I say hello.'
text = text.lower() # 소문자로 대체
text = text.replace('.',' .') # .도 단어로 취급
print(text) # 단어를 띄어쓰기로 구분한 문장 확인

words = text.split(' ') # 띄어쓰기에 맞게 단어 리스트 작성
print(words) # 단어 리스트 확인

word_to_id = {} # 딕셔너리 : 키값이 인덱스
id_to_word = {} # 딕셔너리 : 키값이 단어

for word in words: #단어 리스트의 각 단어별로
    if word not in word_to_id: # 키값에 단어가 없으면
        new_id = len(word_to_id) # 현재 딕셔너리 인덱스 값 new_id에 넣고
        word_to_id[word] = new_id # 키 : word, 값 : new_id
        id_to_word[new_id] = word # 키 : new_id, 값 : word

print(id_to_word) # 키 : word, 값 : new_id
print(word_to_id) # 키 : new_id, 값 : word

corpus = [word_to_id[w] for w in words] # 말뭉치에 각 words에 해당하는 인덱스 값 넣기
corpus = np.array(corpus) # np array로 변환
print(corpus) # 출력

def preprocess(text): # 위 과정 함수화(text 인자로 받고 전처리 후 각 딕셔너리와 말뭉치를 반환)
    text = text.lower()
    text = text.replace('.',' .')
    words = text.split(' ')
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

print(corpus, word_to_id, id_to_word)