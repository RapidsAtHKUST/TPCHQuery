import json

if __name__ == '__main__':
    score_dict = dict()
    valid_num = 0
    with open('raw/scores-10-10.txt') as ifs:
        lines = ifs.readlines()
        for rank, submit_id, id, time in [line.split()[0:4] for line in lines]:
            if id not in score_dict:
                score_dict[id] = []
            score_dict[id].append((rank, submit_id, id, float(time) / (10 ** 3) if rank != '/' else time))
    with open('raw/score-10-10.txt'.replace('.txt', '.json'), 'w') as ofs:
        ofs.write(json.dumps(score_dict, indent=4))
    for k, v in score_dict.items():
        for x in v:
            if x[0] != '/':
                valid_num += 1
                break
    print(valid_num, len(score_dict.items()))
