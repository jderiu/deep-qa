


def main():
    train = "semeval/task-B-train-plus-dev.tsv"
    test = "semeval/task-B-test2014-twitter.tsv"
    dev = "semeval/twitter-test-gold-B.downloaded.tsv"
    test15 = "semeval/task-B-test2015-twitter.tsv"
    train16 = "semeval/task-A-train-2016.tsv"
    dev2016 = "semeval/task-A-dev-2016.tsv"
    devtest2016 = "semeval/task-A-devtest-2016.tsv"
    test2016 = "semeval/SemEval2016-task4-test.subtask-A.txt"

    files = [
        (train,3),
        (dev,3),
        (test,3),
        (test15,3),
        (train16,2),
        (dev2016,2),
        (devtest2016,2),
        (test2016,2)
    ]

    for (fname,pos) in files:
        f = open(fname,'r')
        lines = f.readlines()
        f.close()
        print len(lines)
        lines = filter(lambda line: not line.split('\t')[pos] == 'Not Available\n',lines)
        f = open(fname,'w')
        for line in lines:
            f.write(line)
        f.close()

if __name__ == '__main__':
    main()