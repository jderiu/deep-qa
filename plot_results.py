import cPickle
import matplotlib.pyplot as plt
import numpy as np


def main():
    resmap = cPickle.load(open('sup_res/supervised_results_L2T85Wcustom.p','rb'))

    test_2016n = 'Test 2016'
    test_2015n = 'Test 2015'
    test_2014n = 'Test 2014'
    test_2013n = 'Test 2013'
    test_2014ljn = 'Test 2014 LiveJournal'
    test_2014srcn = 'Test 2014 Sarcasm'
    test_2013_smsn = 'Test 2013 SMS'

    names = [test_2016n,test_2015n,test_2013_smsn,test_2014srcn]
    colors = ['b','g','r','c','m','y','k']

    for (k,c) in zip(resmap.keys(),colors):
        res_k = resmap[k]

        mx = max(res_k)
        idx = resmap[k].index(mx)
        print k,':\t',mx,' at epoch:', (idx + 1)/2.0 , ' mean:', sum(res_k[10:])/len(res_k[10:]), ' std:',np.std(res_k[10:])
        x = np.linspace(start=0.5,stop=len(resmap[k])/5,num=len(res_k))
        y = np.array(res_k)
        if k in names:
            plt.plot(x,y,label=k,color=c)
            plt.plot([(idx + 1)/5.0],[mx],'o',color=c)

    plt.xlabel('Number of Epochs')
    plt.ylabel('F1 score')
    plt.legend(loc=4, borderaxespad=0.2)
    plt.show()

    vals = []
    names = []
    f_out = open('sup_res/corr.txt','w')
    for k in resmap.keys():
        f_out.write(k + '\t')
        names.append(k)
        vals.append(resmap[k])
    f_out.write('\n')
    corr = np.corrcoef(vals)
    n = corr.shape[0]
    m = corr.shape[1]


    for i in xrange(n):
        f_out.write(names[i] + '\t')
        for j in xrange(m):
            f_out.write('{}\t'.format(corr[i,j]))
        f_out.write('\n')
    f_out.close()

if __name__ == '__main__':
    main()