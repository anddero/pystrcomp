import numpy as np
from nltk.metrics import edit_distance
from scipy.optimize import linear_sum_assignment
import string
import re


def file_lines(s):
    with open(s, 'r', encoding='utf-8') as file:
        return file.readlines()


def out_lines(l, s):
    with open(s, 'w') as file:
        for e in l:
            file.write(str(e) + '\n')


def is_clean(s):
    return all(c in 'abcdefghijklmnopqrstuvwxyz0123456789 ' for c in s)


def is_ignored(s):
    return any(c in '恋きьч叙강œðאםច石タнー花ə츄шþ植空洛æクđ러海砂어你и黄다두윈說йп✞' for c in s)


remove_accents_translator = str.maketrans('ọūēⅰă²śłżùńìÿôğõäöüéőóřáíøòëúêñàèąïžšçşåãαāņćčâýı', 'oueia2slzuniyogoaoueooraiooeuenaeaizscsaaaanccayi')
punct_to_space_translator = str.maketrans(string.punctuation + '’”“®´¿–—′', ' ' * len(string.punctuation) + '         ')


def clean_up(s: str):
    s = s.lower()
    s = s.translate(punct_to_space_translator)
    s = s.translate(remove_accents_translator)
    s = re.sub(' +', ' ', s)
    s = s.strip()
    return s


def get_similarity(str1, str2):
    str1 = clean_up(str1)
    str2 = clean_up(str2)
    # print('cleaned up', str1, str2)

    min_chars = min(len(str1), len(str2))
    max_chars = max(len(str1), len(str2))
    # print('min, max chars', min_chars, max_chars)

    if max_chars == 0:
        return 1

    if min_chars == 0:
        return 0

    if is_ignored(str1) or is_ignored(str2):
        print('Ignored: ', str1, ' OR:', str2)
        return -2

    if not is_clean(str1) or not is_clean(str2):
        print('One of these is not clean after clean-up', str1, str2)
        return -1

    # Split strings into words
    words1 = str1.split(' ')
    words2 = str2.split(' ')
    max_words = max(len(words1), len(words2))

    # Pad with spaces to make sure there's the same number of words
    while len(words1) < max_words:
        words1.append('')
    while len(words2) < max_words:
        words2.append('')

    # Initialize distance matrix
    dist_matrix = np.zeros((max_words, max_words))

    # Calculate distance between all pairs of words
    for i, word1 in enumerate(words1):
        for j, word2 in enumerate(words2):
            dist_matrix[i, j] = edit_distance(word1, word2, 1, True)
    # print('dist_matrix', dist_matrix)

    # Find optimal alignment between words
    row_indices, col_indices = linear_sum_assignment(dist_matrix)
    # print('optimal choice', [(row_indices[i], col_indices[i]) for i in range(len(row_indices))])
    minimal_edit_distance = dist_matrix[row_indices, col_indices].sum()
    # print('minimal edit distance', minimal_edit_distance)

    # Normalize distance by length of longest string
    return 1 - minimal_edit_distance / max_chars


def get_extra_word_count_diff(str1, str2):
    str1 = clean_up(str1)
    str2 = clean_up(str2)

    # Split strings into sets of unique words
    words1 = set(str1.split(' '))
    words2 = set(str2.split(' '))
    intersection = words1.intersection(words2)
    words1 = words1.difference(intersection)
    words2 = words2.difference(intersection)

    return len(words2) - len(words1)


def get_extra_word_count_diff_new(str1, str2):
    str1 = clean_up(str1)
    str2 = clean_up(str2)
    # print('cleaned up', str1, str2)

    max_chars = max(len(str1), len(str2))
    # print('min, max chars', min_chars, max_chars)

    if max_chars == 0:
        return 0

    if is_ignored(str1) or is_ignored(str2):
        print('Ignored: ', str1, ' OR:', str2)
        return -1000

    if not is_clean(str1) or not is_clean(str2):
        print('One of these is not clean after clean-up', str1, str2)
        return -2000

    # Split strings into words
    words1 = str1.split(' ')
    words2 = str2.split(' ')
    max_words = max(len(words1), len(words2))

    # Pad with spaces to make sure there's the same number of words
    while len(words1) < max_words:
        words1.append('')
    while len(words2) < max_words:
        words2.append('')

    # Initialize distance matrix
    dist_matrix = np.zeros((max_words, max_words))

    # Calculate distance between all pairs of words
    for i, word1 in enumerate(words1):  # row
        for j, word2 in enumerate(words2):  # col
            dist_matrix[i, j] = edit_distance(word1, word2, 1, True)
    # print('dist_matrix', dist_matrix)

    # Find optimal alignment between words
    row_indices, col_indices = linear_sum_assignment(dist_matrix)
    optimal_choice = [(row_indices[i], col_indices[i]) for i in range(len(row_indices))]
    row_blank_count = 0
    col_blank_count = 0
    for cell in optimal_choice:
        if words1[cell[0]] == '':
            row_blank_count += 1
        if words2[cell[1]] == '':
            col_blank_count += 1
    if row_blank_count > 0 and col_blank_count > 0:
        raise Exception('words1 and words2 cant both have blanks', words1, words2)

    return row_blank_count - col_blank_count


def get_query_containing_score(query, name):
    query = clean_up(query)
    name = clean_up(name)

    max_chars = max(len(query), len(name))

    if max_chars == 0 or len(query) == 0:
        return 1

    if is_ignored(query) or is_ignored(name):
        print('Ignored: ', query, ' OR:', name)
        return -1000

    if not is_clean(query) or not is_clean(name):
        print('One of these is not clean after clean-up', query, name)
        return -2000

    # Split strings into words
    query_words = query.split(' ')
    name_words = name.split(' ')
    max_words = max(len(query_words), len(name_words))

    # Pad with spaces to make sure there's the same number of words
    while len(query_words) < max_words:
        query_words.append('')
    while len(name_words) < max_words:
        name_words.append('')

    # Initialize distance matrix
    dist_matrix = np.zeros((max_words, max_words))

    # Calculate distance between all pairs of words
    for i, qword in enumerate(query_words):  # row
        for j, nword in enumerate(name_words):  # col
            dist_matrix[i, j] = edit_distance(qword, nword, 1, True)

    # Find optimal alignment between words
    row_indices, col_indices = linear_sum_assignment(dist_matrix)
    optimal_scores = dist_matrix[row_indices, col_indices]
    optimal_choice = [(row_indices[i], col_indices[i]) for i in range(len(row_indices))]
    score = 0
    for (i, cell) in enumerate(optimal_choice):
        qword = query_words[cell[0]]
        nword = name_words[cell[1]]
        if qword == '':
            continue
        if nword == '':
            continue
        diff = optimal_scores[i]
        if diff >= len(qword):
            continue
        score += len(qword) - diff

    return score / len(query.replace(' ', ''))


def scoreScript():
    input1 = file_lines('input1.txt')
    input2 = file_lines('input2.txt')
    if len(input1) != len(input2):
        print('Input files must have the same number of elements, given', len(input1), 'and', len(input2))
        exit(1)
    output = []
    for i in range(len(input1)):
        x = get_similarity(input1[i], input2[i])
        if x == -1:
            print('Negative: ', i)
            exit(1)
        output.append(x)
    out_lines(output, 'output.txt')


def queryContainingScoreScript():
    input_queries = file_lines('input_queries.txt')
    input_names = file_lines('input_names.txt')
    if len(input_queries) != len(input_names):
        print('Input files must have the same number of elements, given', len(input_queries), 'and', len(input_names))
        exit(1)
    output = []
    for i in range(len(input_queries)):
        x = get_query_containing_score(input_queries[i], input_names[i])
        output.append(x)
    out_lines(output, 'output.txt')


def extraWordCountDiffScript():
    input1 = file_lines('input1.txt')
    input2 = file_lines('input2.txt')
    if len(input1) != len(input2):
        print('Input files must have the same number of elements, given', len(input1), 'and', len(input2))
        exit(1)
    output = []
    for i in range(len(input1)):
        x = get_extra_word_count_diff_new(input1[i], input2[i])
        output.append(x)
    out_lines(output, 'output.txt')


def scoreScriptHyphenComma():
    input_queries = file_lines('input_queries.txt')
    input_names = file_lines('input_names.txt')
    if len(input_queries) != len(input_names):
        print('Input files must have the same number of elements, given', len(input_queries), 'and', len(input_names))
        exit(1)
    output = []
    for i in range(len(input_queries)):
        split_by_hyphen_and_comma = input_names[i].replace(',', '-').split('-')
        max_x = -10
        for j in range(len(split_by_hyphen_and_comma)):
            joined = '-'.join(split_by_hyphen_and_comma[0:(j+1)])
            # print('joined: ', joined)
            x = get_similarity(input_queries[i], joined)
            if x > max_x:
                max_x = x
        if max_x == -1:
            print('Negative: ', i)
            exit(1)
        output.append(max_x)
    out_lines(output, 'output.txt')


def scoreScriptNameArtist():
    input_queries = file_lines('input_queries.txt')
    input_names = file_lines('input_names.txt')
    if len(input_queries) != len(input_names):
        print('Input files must have the same number of elements, given', len(input_queries), 'and', len(input_names))
        exit(1)
    output = []
    for i in range(len(input_queries)):
        name = input_names[i]
        query = input_queries[i]
        if name.count('-') != 1:
            output.append(-13)
            continue
        [name, artists] = name.split('-')
        artists = artists.split(',')
        max_x = -200
        for artist in artists:
            tokens = clean_up(artist).split(' ')
            for token in tokens:
                joined = name + token
                x = get_similarity(query, joined)
                if x > max_x:
                    max_x = x
            x = get_similarity(query, artist)
            if x > max_x:
                max_x = x
        if max_x == -1:
            print('Negative: ', i)
            exit(1)
        output.append(max_x)
    out_lines(output, 'output.txt')


def cleanUpScript():
    inlines = file_lines('input.txt')
    output = []
    for line in inlines:
        output.append(clean_up(line))
    out_lines(output, 'output.txt')


if __name__ == '__main__':
    queryContainingScoreScript()
