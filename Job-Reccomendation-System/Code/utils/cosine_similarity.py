import numpy as np
import math
def cosine_similarity(vec1, vec2):
    vec1_list = list(vec1.values())
    vec2_list = list(vec2.values())
    dot_product = np.dot(vec1_list, vec2_list)
    mag_vec1 = math.sqrt(sum(x**2 for x in vec1_list))
    mag_vec2 = math.sqrt(sum(x**2 for x in vec2_list))

    if mag_vec1 == 0 or mag_vec2 == 0:
        return 0  

    similarity = dot_product / (mag_vec1 * mag_vec2)
    return similarity

def cal_cosine_similarity(query_vec, doc_vecs):
    return [cosine_similarity(query_vec, doc_vec) for doc_vec in doc_vecs]